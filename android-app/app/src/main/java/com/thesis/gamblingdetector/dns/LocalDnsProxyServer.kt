package com.thesis.gamblingdetector.dns

import android.net.VpnService
import android.util.Log
import com.thesis.gamblingdetector.data.DetectionLogger
import com.thesis.gamblingdetector.ml.DomainClassifier
import com.thesis.gamblingdetector.vpn.DnsPacketParser
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.FileDescriptor
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress

/**
 * In-process DNS proxy that sits between the TUN interface and the real
 * upstream DNS resolver.
 *
 * HOW IT WORKS (DNS-only VPN route design):
 * ─────────────────────────────────────────
 * Because GamblingVpnService now only routes 10.111.222.1/32 (our fake
 * DNS server address) through the TUN interface — NOT the full default
 * route — ALL regular internet traffic (TCP connections to Google, YouTube,
 * CDNs, etc.) flows normally over the physical network interface and is
 * never touched by this service.
 *
 * Only DNS queries reach the TUN, because the system resolver is pointed at
 * 10.111.222.1 (our fake DNS address), which is routed through the TUN.
 * This class:
 *
 *   1. Reads raw IPv4/UDP packets from the TUN fd (each is a DNS query).
 *   2. Parses the query with [DnsPacketParser.tryParseDnsQuery].
 *   3. Classifies the queried domain with [DomainClassifier].
 *   4. If BLOCKED  → synthesises an NXDOMAIN response and writes it back
 *                    into the TUN; the requesting app sees "domain not found".
 *   5. If ALLOWED  → opens a DatagramSocket protected from the VPN (so it
 *                    uses the real network, not the TUN), forwards the raw
 *                    DNS query to the upstream resolver (default: 8.8.8.8),
 *                    receives the real answer, and relays it back into the
 *                    TUN so the requesting app gets a real DNS response.
 *
 * LOGGING:
 *   - Every decision is logged at Log.d/i level under [TAG].
 *   - Blocked domains are logged at Log.i with a clear "BLOCKED" prefix.
 *   - Allowed domains are logged at Log.d with score for debugging.
 *   - Parse/IO errors are logged at Log.w so they are visible in logcat
 *     without flooding the log under normal operation.
 *
 * Run [start] once after the TUN is established; call [stop] to cleanly
 * cancel the coroutine and release resources.
 */
class LocalDnsProxyServer(
    /** The VpnService instance — needed to call protect() on upstream sockets. */
    private val vpnService: VpnService,
    /** TUN file descriptor as returned by Builder.establish(). */
    private val tunFd: FileDescriptor,
    private val classifier: DomainClassifier,
    private val logger: DetectionLogger,
    /** Upstream DNS resolver addresses in priority order (device's own DNS first, then public fallbacks). */
    private val upstreamDnsServers: List<String> = listOf("8.8.8.8", "1.1.1.1", "8.8.4.4"),
    private val upstreamDnsPort: Int = DNS_PORT,
) {

    companion object {
        private const val TAG = "LocalDnsProxyServer"
        private const val DNS_PORT = 53

        /**
         * Buffer size for DNS responses and TUN packets (supports EDNS0 up to 4096 B).
         */
        private const val DNS_BUFFER_SIZE = 4096

        /**
         * Timeout (ms) per upstream DNS server query.
         */
        private const val UPSTREAM_SOCKET_TIMEOUT_MS = 1_500

        /**
         * Maximum number of consecutive read errors before the loop gives up.
         * Prevents a tight error-spin that would drain the battery.
         */
        private const val MAX_CONSECUTIVE_ERRORS = 20
    }

    @Volatile private var loopJob: Job? = null

    /**
     * Starts the packet-processing coroutine on [scope].
     * Safe to call only once; subsequent calls are ignored if already running.
     */
    fun start(scope: CoroutineScope) {
        if (loopJob?.isActive == true) {
            Log.w(TAG, "start() called but loop is already running — ignored")
            return
        }
        Log.i(TAG, "Starting DNS proxy loop (upstreams=$upstreamDnsServers, port=$upstreamDnsPort)")
        loopJob = scope.launch(Dispatchers.IO) { runLoop() }
    }

    /** Cancels the processing coroutine. Safe to call from any thread. */
    fun stop() {
        Log.i(TAG, "Stopping DNS proxy loop")
        loopJob?.cancel()
        loopJob = null
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal
    // ─────────────────────────────────────────────────────────────────────────

    private suspend fun runLoop() {
        val inputStream  = FileInputStream(tunFd)
        val outputStream = FileOutputStream(tunFd)
        val buffer = ByteArray(DNS_BUFFER_SIZE)
        var consecutiveErrors = 0

        Log.d(TAG, "Packet loop started (blocking IO on Dispatchers.IO)")

        while (currentCoroutineContext().isActive) {
            try {
                // withContext(Dispatchers.IO) ensures the blocking read() runs
                // on the IO thread pool and does not block the coroutine
                // dispatcher in a way that starves other coroutines.
                // With setBlocking(true), read() suspends here until a packet
                // arrives from the TUN interface.
                val length = withContext(Dispatchers.IO) { inputStream.read(buffer) }

                when {
                    length == -1 -> {
                        // True EOF: the TUN file descriptor was closed.
                        // This is the expected signal when stopVpn() closes
                        // the ParcelFileDescriptor, so exit cleanly.
                        Log.i(TAG, "TUN fd closed (EOF) — exiting packet loop")
                        break
                    }
                    length == 0 -> {
                        // Should not happen with setBlocking(true), but guard
                        // defensively. Do NOT break — just go back to read().
                        Log.v(TAG, "TUN read returned 0 bytes — retrying")
                        continue
                    }
                    else -> {
                        consecutiveErrors = 0 // reset on successful read
                    }
                }

                val query = DnsPacketParser.tryParseDnsQuery(buffer, length)
                if (query == null) {
                    // Normal: non-DNS packet arrived on TUN. With a DNS-only
                    // route this should be rare (only ICMP to the fake DNS
                    // address). Skip silently.
                    Log.v(TAG, "Skipping non-DNS packet (length=$length)")
                    continue
                }

                Log.d(TAG, "DNS query received: domain='${query.domain}'")
                processQuery(query, outputStream)

            } catch (e: CancellationException) {
                // Coroutine was cancelled (e.g. stop() was called).
                // Re-throw so the coroutine framework can clean up correctly.
                Log.d(TAG, "Coroutine cancelled — shutting down loop")
                throw e
            } catch (e: Exception) {
                consecutiveErrors++
                Log.w(TAG, "Error in packet loop (consecutive=$consecutiveErrors): ${e.javaClass.simpleName}: ${e.message}")
                if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
                    Log.e(TAG, "Too many consecutive errors — stopping loop to avoid spin")
                    break
                }
            }
        }

        Log.d(TAG, "Packet loop exited")
        try { inputStream.close()  } catch (_: Exception) {}
        try { outputStream.close() } catch (_: Exception) {}
    }

    private val SINKHOLE_IP = byteArrayOf(0, 0, 0, 0)

    /**
     * Classifies [query]'s domain and either blocks it (Sinkhole 0.0.0.0 / NXDOMAIN) or
     * forwards it to the upstream resolver and relays the real answer.
     */
    private fun processQuery(
        query: DnsPacketParser.ParsedDnsQuery,
        outputStream: FileOutputStream,
    ) {
        val rawDomain = query.domain
        val normalizedDomain = DomainClassifier.normalize(rawDomain)

        val prediction = try {
            classifier.classify(rawDomain)
        } catch (e: Exception) {
            Log.e(TAG, "Classifier threw exception for domain='$rawDomain': ${e.message}", e)
            // FAIL-SECURE: If prediction crashes, check if domain contains gambling keywords.
            // Do NOT silently allow suspicious domains!
            val containsGamblingKeywords = rawDomain.contains("slot") ||
                rawDomain.contains("judi") ||
                rawDomain.contains("bet") ||
                rawDomain.contains("win") ||
                rawDomain.contains("link") ||
                rawDomain.contains("gacor") ||
                rawDomain.contains("togel")

            if (containsGamblingKeywords) {
                Log.w(TAG, "FAIL-SECURE: Blocking suspicious domain '$rawDomain' after classifier failure")
                DomainClassifier.Prediction(
                    domain = normalizedDomain,
                    score = 0.99,
                    isGambling = true,
                    note = "Blocked via fail-secure fallback"
                )
            } else {
                Log.w(TAG, "Allowing non-suspicious domain '$rawDomain' after classifier error")
                DomainClassifier.Prediction(
                    domain = normalizedDomain,
                    score = 0.0,
                    isGambling = false,
                    note = "Allowed via fail-safe fallback"
                )
            }
        }

        val domainToLog = if (prediction.domain.isNotEmpty()) prediction.domain else rawDomain

        if (prediction.isGambling) {
            Log.i(
                TAG,
                "BLOCKED domain='$domainToLog' (raw='$rawDomain') " +
                    "score=${"%.4f".format(prediction.score)} " +
                    "reason='${prediction.note}' -> Responding with Sinkhole 0.0.0.0"
            )
            logger.logDetection(domainToLog, "BLOCKED", prediction.score)

            // Respond with DNS Sinkhole (0.0.0.0)
            // This stops Android resolver and browsers from retrying secondary or physical DNS.
            try {
                val sinkholeResponse = DnsPacketParser.buildSinkholeResponse(query, SINKHOLE_IP)
                writeToTun(outputStream, sinkholeResponse, "Sinkhole 0.0.0.0 for '$rawDomain'")
            } catch (e: Exception) {
                Log.w(TAG, "Sinkhole build failed, falling back to NXDOMAIN: ${e.message}")
                val nxDomain = DnsPacketParser.buildNxDomainResponse(query)
                writeToTun(outputStream, nxDomain, "NXDOMAIN for '$rawDomain'")
            }
        } else {
            Log.d(
                TAG,
                "ALLOWED domain='$domainToLog' " +
                    "score=${"%.4f".format(prediction.score)} -> Forwarding to upstream"
            )
            logger.logDetection(domainToLog, "ALLOWED", prediction.score)
            forwardToUpstreamAndRelay(query, outputStream)
        }
    }

    /**
     * Opens a UDP socket protected from the VPN (so it uses the real
     * network interface), sends the DNS query to [upstreamDns], waits for
     * the response, and writes it back into the TUN fd wrapped in the
     * correct IPv4/UDP headers so the requesting app receives a real answer.
     */
    /**
     * Forwards the exact raw DNS query to the upstream resolver list with failover.
     * Protected UDP socket ensures the query goes over the physical network interface.
     * The response is wrapped in IPv4/UDP and written back to the TUN fd.
     */
    private fun forwardToUpstreamAndRelay(
        query: DnsPacketParser.ParsedDnsQuery,
        outputStream: FileOutputStream,
    ) {
        val rawDnsQuery = query.rawDnsBytes
        val responseBuffer = ByteArray(DNS_BUFFER_SIZE)

        for (upstreamHost in upstreamDnsServers) {
            var socket: DatagramSocket? = null
            try {
                socket = DatagramSocket()
                // CRITICAL: protect() exempts this socket from the VPN tunnel.
                // Without this call the outbound DNS request would be routed back
                // into the TUN — creating an infinite loop — instead of going out
                // over the physical network interface.
                if (!vpnService.protect(socket)) {
                    Log.w(TAG, "protect() returned false for upstream socket ($upstreamHost) — skipping")
                    continue
                }
                socket.soTimeout = UPSTREAM_SOCKET_TIMEOUT_MS

                val upstream = InetAddress.getByName(upstreamHost)
                val sendPacket = DatagramPacket(rawDnsQuery, rawDnsQuery.size, upstream, upstreamDnsPort)
                socket.send(sendPacket)

                val receivePacket = DatagramPacket(responseBuffer, responseBuffer.size)
                socket.receive(receivePacket)

                val responseBytes = responseBuffer.copyOf(receivePacket.length)
                val relayPacket = DnsPacketParser.wrapDnsResponseForRelay(query, responseBytes)
                writeToTun(outputStream, relayPacket, "upstream response for '${query.domain}' via $upstreamHost")
                Log.d(TAG, "Relayed response for '${query.domain}' from $upstreamHost (${receivePacket.length} bytes)")
                return // Successfully resolved and relayed
            } catch (e: Exception) {
                Log.w(
                    TAG,
                    "Failed upstream $upstreamHost for '${query.domain}': ${e.javaClass.simpleName}: ${e.message}"
                )
            } finally {
                try { socket?.close() } catch (_: Exception) {}
            }
        }

        // If all upstream DNS servers failed (or timed out), send NXDOMAIN as fallback
        Log.e(TAG, "All upstreams failed for '${query.domain}' — sending fallback NXDOMAIN")
        try {
            val nxDomain = DnsPacketParser.buildNxDomainResponse(query)
            writeToTun(outputStream, nxDomain, "fallback NXDOMAIN (all upstreams failed) for '${query.domain}'")
        } catch (inner: Exception) {
            Log.e(TAG, "Also failed to write fallback NXDOMAIN: ${inner.message}")
        }
    }

    private fun writeToTun(outputStream: FileOutputStream, packet: ByteArray, description: String) {
        try {
            outputStream.write(packet)
            Log.v(TAG, "Wrote $description (${packet.size} bytes) to TUN")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to write $description to TUN: ${e.message}")
        }
    }
}
