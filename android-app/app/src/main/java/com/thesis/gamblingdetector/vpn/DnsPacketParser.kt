package com.thesis.gamblingdetector.vpn

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Minimal IPv4 / UDP / DNS parsing & response-building, purpose-built for
 * this VPN's DNS-filtering use case (NOT a general-purpose network stack).
 *
 * Design note: we only need to understand DNS *queries* well enough to
 * extract the queried domain name, and only need to *build* a valid DNS
 * *response* (NXDOMAIN or sinkhole A-record) well enough for the Android
 * DNS resolver on the requesting app to accept it. We deliberately do not
 * implement general IPv4 option handling, IPv6, fragmentation, TCP DNS
 * (RFC 7766), or EDNS0 parsing — documented as known limitations.
 */
object DnsPacketParser {

    private const val PROTOCOL_UDP = 17
    private const val DNS_PORT = 53

    data class ParsedDnsQuery(
        val domain: String,
        val transactionId: ByteArray,
        val questionSectionRaw: ByteArray, // QNAME+QTYPE+QCLASS, needed to echo back
        val sourceAddress: ByteArray,      // original requester IP (becomes dest of response)
        val sourcePort: Int,
        val destAddress: ByteArray,        // our fake DNS server IP (becomes source of response)
        val destPort: Int,
        val ipHeaderLength: Int,
        val rawDnsBytes: ByteArray,        // complete raw DNS query payload from client
    )

    /**
     * Attempts to parse [packet] (length [length]) as an IPv4/UDP/DNS query.
     * Returns null if it's not a DNS query we can/should handle (e.g. not
     * UDP, not port 53, malformed, or a DNS response rather than a query).
     */
    fun tryParseDnsQuery(packet: ByteArray, length: Int): ParsedDnsQuery? {
        if (length < 20) return null
        val versionAndIhl = packet[0].toInt() and 0xFF
        val version = versionAndIhl shr 4
        if (version != 4) return null // IPv6 not handled in this prototype

        val ihl = (versionAndIhl and 0x0F) * 4
        if (length < ihl + 8) return null

        val protocol = packet[9].toInt() and 0xFF
        if (protocol != PROTOCOL_UDP) return null

        val srcAddr = packet.copyOfRange(12, 16)
        val dstAddr = packet.copyOfRange(16, 20)

        val udpOffset = ihl
        val srcPort = readUInt16(packet, udpOffset)
        val dstPort = readUInt16(packet, udpOffset + 2)
        if (dstPort != DNS_PORT) return null

        val udpLength = readUInt16(packet, udpOffset + 4)
        val dnsOffset = udpOffset + 8
        if (dnsOffset >= length) return null

        val flags = readUInt16(packet, dnsOffset + 2)
        val isResponse = (flags shr 15) and 0x1 == 1
        if (isResponse) return null // we only care about queries here

        val qdCount = readUInt16(packet, dnsOffset + 4)
        if (qdCount < 1) return null

        val transactionId = packet.copyOfRange(dnsOffset, dnsOffset + 2)

        // Parse QNAME starting at dnsOffset + 12
        val nameStart = dnsOffset + 12
        val sb = StringBuilder()
        var pos = nameStart
        while (pos < length) {
            val labelLen = packet[pos].toInt() and 0xFF
            if (labelLen == 0) {
                pos += 1
                break
            }
            // Compression pointers are not expected in a query's own QNAME
            // in practice; bail out defensively if encountered.
            if (labelLen and 0xC0 == 0xC0) return null
            pos += 1
            if (pos + labelLen > length) return null
            if (sb.isNotEmpty()) sb.append('.')
            sb.append(String(packet, pos, labelLen, Charsets.US_ASCII))
            pos += labelLen
        }
        if (sb.isEmpty()) return null

        // QTYPE + QCLASS follow (4 bytes)
        val questionEnd = pos + 4
        if (questionEnd > length) return null
        val questionSectionRaw = packet.copyOfRange(nameStart, questionEnd)

        val rawDnsBytes = packet.copyOfRange(dnsOffset, length)

        return ParsedDnsQuery(
            domain = sb.toString().lowercase(),
            transactionId = transactionId,
            questionSectionRaw = questionSectionRaw,
            sourceAddress = srcAddr,
            sourcePort = srcPort,
            destAddress = dstAddr,
            destPort = dstPort,
            ipHeaderLength = ihl,
            rawDnsBytes = rawDnsBytes,
        )
    }

    /**
     * Builds a complete IPv4/UDP/DNS packet that answers [query] with
     * NXDOMAIN (RCODE=3, no answer records). This is sent back into the
     * TUN interface as if it came from the (fake) DNS server, so the
     * requesting app on-device sees a normal "domain does not exist"
     * failure — the same experience as a standard DNS sinkhole.
     */
    fun buildNxDomainResponse(query: ParsedDnsQuery): ByteArray {
        // --- DNS section ---
        val dns = ByteBuffer.allocate(12 + query.questionSectionRaw.size)
        dns.put(query.transactionId)
        // flags: QR=1 (response), Opcode=0, AA=0, TC=0, RD=1, RA=1, Z=0, RCODE=3 (NXDOMAIN)
        val flags = 0x8180 or 0x0003
        dns.putShort(flags.toShort())
        dns.putShort(1) // QDCOUNT
        dns.putShort(0) // ANCOUNT
        dns.putShort(0) // NSCOUNT
        dns.putShort(0) // ARCOUNT
        dns.put(query.questionSectionRaw)
        val dnsBytes = dns.array()

        return wrapInUdpIpv4(
            payload = dnsBytes,
            srcAddr = query.destAddress,   // our fake DNS server becomes the source
            srcPort = query.destPort,
            dstAddr = query.sourceAddress, // original requester becomes destination
            dstPort = query.sourcePort,
        )
    }

    /**
     * Alternative to NXDOMAIN: answer with a real A record pointing to a
     * "sinkhole" IP (e.g. a local page explaining the block, or 0.0.0.0).
     * Useful if you want the browser to show a custom "blocked" landing
     * page instead of a generic DNS error.
     */
    fun buildSinkholeResponse(query: ParsedDnsQuery, sinkholeIp: ByteArray): ByteArray {
        require(sinkholeIp.size == 4) { "sinkholeIp must be a 4-byte IPv4 address" }

        val answer = ByteBuffer.allocate(16)
        answer.putShort(0xC00C.toShort()) // pointer to name at offset 12 (the question)
        answer.putShort(1) // TYPE A
        answer.putShort(1) // CLASS IN
        answer.putInt(60)  // TTL seconds
        answer.putShort(4) // RDLENGTH
        answer.put(sinkholeIp)
        val answerBytes = answer.array()

        val dns = ByteBuffer.allocate(12 + query.questionSectionRaw.size + answerBytes.size)
        dns.put(query.transactionId)
        val flags = 0x8180 // QR=1, RD=1, RA=1, RCODE=0
        dns.putShort(flags.toShort())
        dns.putShort(1) // QDCOUNT
        dns.putShort(1) // ANCOUNT
        dns.putShort(0) // NSCOUNT
        dns.putShort(0) // ARCOUNT
        dns.put(query.questionSectionRaw)
        dns.put(answerBytes)
        val dnsBytes = dns.array()

        return wrapInUdpIpv4(
            payload = dnsBytes,
            srcAddr = query.destAddress,
            srcPort = query.destPort,
            dstAddr = query.sourceAddress,
            dstPort = query.sourcePort,
        )
    }

    /**
     * Wraps a raw DNS *response* payload received from a real upstream
     * resolver (e.g. 8.8.8.8) into an IPv4/UDP packet addressed back to
     * the original requester, so it can be written into the TUN
     * interface and delivered to the app that made the query — this is
     * the "allowed" path's relay step (as opposed to buildNxDomainResponse
     * / buildSinkholeResponse, which synthesize a response locally for
     * the "blocked" path).
     */
    fun wrapDnsResponseForRelay(query: ParsedDnsQuery, upstreamDnsResponseBytes: ByteArray): ByteArray {
        return wrapInUdpIpv4(
            payload = upstreamDnsResponseBytes,
            srcAddr = query.destAddress,   // our fake DNS server address (as seen by the app)
            srcPort = query.destPort,
            dstAddr = query.sourceAddress,
            dstPort = query.sourcePort,
        )
    }

    /**
     * Wraps [payload] (already-complete DNS message bytes) into a UDP
     * datagram, then an IPv4 packet, with valid IPv4 header and UDP checksums.
     */
    private fun wrapInUdpIpv4(
        payload: ByteArray,
        srcAddr: ByteArray,
        srcPort: Int,
        dstAddr: ByteArray,
        dstPort: Int,
    ): ByteArray {
        val udpLength = 8 + payload.size
        val totalLength = 20 + udpLength

        val buffer = ByteBuffer.allocate(totalLength)
        buffer.order(ByteOrder.BIG_ENDIAN)

        // --- IPv4 header ---
        buffer.put((0x45).toByte())      // version=4, IHL=5 (20 bytes, no options)
        buffer.put(0x00.toByte())        // DSCP/ECN
        buffer.putShort(totalLength.toShort())
        buffer.putShort(0x0000)          // identification
        buffer.putShort(0x4000.toShort())// flags=DF, fragment offset=0
        buffer.put(64.toByte())          // TTL
        buffer.put(PROTOCOL_UDP.toByte())
        val ipChecksumPos = buffer.position()
        buffer.putShort(0x0000)          // checksum placeholder
        buffer.put(srcAddr)
        buffer.put(dstAddr)

        // --- UDP header ---
        val udpStartPos = buffer.position()
        buffer.putShort(srcPort.toShort())
        buffer.putShort(dstPort.toShort())
        buffer.putShort(udpLength.toShort())
        val udpChecksumPos = buffer.position()
        buffer.putShort(0x0000)          // UDP checksum placeholder

        // --- Payload (DNS message) ---
        buffer.put(payload)

        val bytes = buffer.array()

        // Compute IPv4 header checksum
        val ipChecksum = computeIpv4HeaderChecksum(bytes, 0, 20)
        bytes[ipChecksumPos] = (ipChecksum shr 8).toByte()
        bytes[ipChecksumPos + 1] = (ipChecksum and 0xFF).toByte()

        // Compute UDP checksum (RFC 768)
        val udpChecksum = computeUdpChecksum(srcAddr, dstAddr, bytes, udpStartPos, udpLength)
        val checkInt = udpChecksum.toInt() and 0xFFFF
        bytes[udpChecksumPos] = (checkInt shr 8).toByte()
        bytes[udpChecksumPos + 1] = (checkInt and 0xFF).toByte()

        return bytes
    }

    private fun computeUdpChecksum(
        srcAddr: ByteArray,
        dstAddr: ByteArray,
        data: ByteArray,
        udpOffset: Int,
        udpLength: Int,
    ): Short {
        var sum = 0
        // IPv4 Pseudo-Header
        sum += ((srcAddr[0].toInt() and 0xFF) shl 8) or (srcAddr[1].toInt() and 0xFF)
        sum += ((srcAddr[2].toInt() and 0xFF) shl 8) or (srcAddr[3].toInt() and 0xFF)
        sum += ((dstAddr[0].toInt() and 0xFF) shl 8) or (dstAddr[1].toInt() and 0xFF)
        sum += ((dstAddr[2].toInt() and 0xFF) shl 8) or (dstAddr[3].toInt() and 0xFF)
        sum += PROTOCOL_UDP
        sum += udpLength

        // UDP Header and Data
        var i = udpOffset
        val end = udpOffset + udpLength
        while (i < end - 1) {
            sum += ((data[i].toInt() and 0xFF) shl 8) or (data[i + 1].toInt() and 0xFF)
            i += 2
        }
        if (i < end) {
            sum += (data[i].toInt() and 0xFF) shl 8
        }

        while (sum shr 16 != 0) {
            sum = (sum and 0xFFFF) + (sum shr 16)
        }
        val checksum = sum.inv() and 0xFFFF
        return (if (checksum == 0) 0xFFFF else checksum).toShort()
    }

    private fun computeIpv4HeaderChecksum(data: ByteArray, offset: Int, length: Int): Int {
        var sum = 0
        var i = offset
        while (i < offset + length) {
            val word = (readUInt8(data, i) shl 8) or readUInt8(data, i + 1)
            sum += word
            i += 2
        }
        while (sum shr 16 != 0) {
            sum = (sum and 0xFFFF) + (sum shr 16)
        }
        return sum.inv() and 0xFFFF
    }

    private fun readUInt8(data: ByteArray, index: Int): Int = data[index].toInt() and 0xFF

    private fun readUInt16(data: ByteArray, offset: Int): Int {
        return ((data[offset].toInt() and 0xFF) shl 8) or (data[offset + 1].toInt() and 0xFF)
    }
}
