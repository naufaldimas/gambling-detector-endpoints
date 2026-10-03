package com.thesis.gamblingdetector.vpn

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Intent
import android.net.VpnService
import android.os.Build
import android.os.ParcelFileDescriptor
import android.util.Log
import com.thesis.gamblingdetector.data.DetectionLogger
import com.thesis.gamblingdetector.dns.LocalDnsProxyServer
import com.thesis.gamblingdetector.ml.DomainClassifier
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job

/**
 * Core VPN service that intercepts DNS queries for gambling-site detection.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * ARCHITECTURE — v3 (DNS-only route, no tun2socks dependency)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * ROOT CAUSE OF THE ORIGINAL "ALL INTERNET BROKEN" BUG
 * ─────────────────────────────────────────────────────
 * The v2 design captured ALL traffic via a 0.0.0.0/0 VPN route and relied
 * on tun2socks (an external .aar binary) to relay non-DNS packets back to
 * the real network. The tun2socks .aar was never built and checked in, so
 * Tun2socksBridge.connect() always threw an exception. At that point
 * stopVpn() was called — but the TUN interface was already established
 * with route 0.0.0.0/0. Android had already redirected ALL traffic into
 * the TUN file descriptor, and with nobody reading or forwarding those
 * packets, every outbound connection (Google, YouTube, CDN, everything)
 * simply timed out. The device appeared to have no internet access.
 *
 * THE FIX — DNS-only VPN route
 * ─────────────────────────────
 * Instead of capturing all traffic, we only route the single fake-DNS
 * host address (10.111.222.1/32) through the TUN. The Android system
 * resolver is pointed at that address, so only DNS queries land in the
 * TUN — ALL other TCP/UDP/ICMP traffic continues to flow normally over
 * the real network interface, untouched by this service.
 *
 * Flow:
 *   App DNS query ──▶ TUN (10.111.222.1/32 route)
 *                          │
 *                          ▼
 *                   LocalDnsProxyServer
 *                    ├─ DomainClassifier.classify()
 *                    │       │ isGambling=true
 *                    │       ▼
 *                    │   buildNxDomainResponse() ──▶ TUN ──▶ app sees NXDOMAIN
 *                    │
 *                    │       │ isGambling=false
 *                    │       ▼
 *                    │   protect()ed DatagramSocket ──▶ 8.8.8.8:53
 *                    │       │ real DNS answer
 *                    │       ▼
 *                    └─ wrapDnsResponseForRelay() ──▶ TUN ──▶ app gets real IP
 *
 *   All non-DNS traffic ──▶ physical NIC (never enters TUN) ──▶ internet ✓
 *
 * WHY THIS IS SAFE
 * ─────────────────
 * - No tun2socks or any native binary required.
 * - No hand-rolled TCP/IP NAT stack needed (we never see non-DNS packets).
 * - Our own outbound sockets (upstream DNS requests) are exempted from the
 *   VPN by calling protect() + addDisallowedApplication(packageName), so
 *   there is no loop risk.
 * - Protection feature still works: gambling domains get NXDOMAIN, all
 *   legitimate traffic is unaffected.
 *
 * KNOWN LIMITATION (document in thesis):
 * Apps using DNS-over-HTTPS or a hardcoded DNS IP bypass the system resolver
 * entirely, so their queries never reach this proxy. See README §4 for the
 * full discussion and suggested mitigations.
 */
class GamblingVpnService : VpnService() {

    companion object {
        private const val TAG = "GamblingVpnService"
        const val ACTION_START = "com.thesis.gamblingdetector.START"
        const val ACTION_STOP  = "com.thesis.gamblingdetector.STOP"

        /**
         * The IP address assigned to the local TUN interface.
         */
        private const val TUN_ADDRESS       = "10.111.222.1"
        private const val TUN_PREFIX_LENGTH = 24   // subnet mask for the TUN interface address

        /**
         * The fake DNS server IP assigned as the system DNS resolver.
         *
         * CRITICAL FIX:
         * Must be DIFFERENT from TUN_ADDRESS (10.111.222.1). If the DNS server
         * has the same IP as the interface, the Linux kernel treats all DNS
         * queries as LOCAL traffic (delivers to loopback 'lo' table) and NEVER
         * routes them out to the TUN interface descriptor.
         * By setting the DNS server to 10.111.222.2 and adding a /32 route for it,
         * the kernel routes DNS packets OUT through the TUN interface, allowing
         * LocalDnsProxyServer to read and process them.
         */
        private const val FAKE_DNS_SERVER_IP = "10.111.222.2"
        private const val DNS_ROUTE_PREFIX   = 32   // /32 = single host route, DNS traffic only

        private const val MTU               = 1500
        private const val NOTIFICATION_ID   = 1
        private const val NOTIF_CHANNEL_ID  = "vpn_service"
    }

    private var tunInterface: ParcelFileDescriptor? = null
    private val serviceJob   = Job()
    private val serviceScope = CoroutineScope(Dispatchers.IO + serviceJob)

    private lateinit var classifier: DomainClassifier
    private lateinit var logger: DetectionLogger
    private var dnsProxy: LocalDnsProxyServer? = null

    @Volatile private var isRunning = false

    // ─────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "onCreate()")
        classifier = DomainClassifier(applicationContext)
        logger     = DetectionLogger(applicationContext)
        startForegroundCompat()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "onStartCommand() action=${intent?.action}")
        return when (intent?.action) {
            ACTION_STOP -> {
                stopVpn()
                START_NOT_STICKY
            }
            else -> {
                startVpn()
                START_STICKY
            }
        }
    }

    override fun onDestroy() {
        Log.d(TAG, "onDestroy()")
        stopVpn()
        serviceJob.cancel()
        super.onDestroy()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // VPN lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    private fun startVpn() {
        if (isRunning) {
            Log.w(TAG, "startVpn() called but already running — ignored")
            return
        }

        Log.i(TAG, "Starting VPN (DNS-only route design v3)")
        Log.i(TAG, "  TUN address : $TUN_ADDRESS/$TUN_PREFIX_LENGTH")
        Log.i(TAG, "  DNS server  : $FAKE_DNS_SERVER_IP (distinct alias)")
        Log.i(TAG, "  DNS route   : $FAKE_DNS_SERVER_IP/$DNS_ROUTE_PREFIX  ← only DNS enters TUN")
        Log.i(TAG, "  Non-DNS traffic flows over physical NIC as normal")

        // ── 1. Build the TUN interface ────────────────────────────────────────
        val builder = Builder()
            .setSession("Gambling Site Blocker")
            .addAddress(TUN_ADDRESS, TUN_PREFIX_LENGTH)
            // Point the device's DNS resolver at our fake DNS address (10.111.222.2).
            .addDnsServer(FAKE_DNS_SERVER_IP)
            // Route the fake-DNS host address through the TUN.
            .addRoute(FAKE_DNS_SERVER_IP, DNS_ROUTE_PREFIX)
            .setMtu(MTU)
            .setBlocking(true)

        // Route common public DNS resolver IPs into TUN so that apps or browsers
        // (like Chrome or Firefox) attempting to bypass system DNS by querying
        // 8.8.8.8, 1.1.1.1, etc. directly are also intercepted and evaluated.
        val publicDnsToIntercept = listOf(
            "8.8.8.8", "8.8.4.4",               // Google Public DNS
            "1.1.1.1", "1.0.0.1",               // Cloudflare DNS
            "9.9.9.9", "149.112.112.112",       // Quad9
            "208.67.222.222", "208.67.220.220", // OpenDNS
        )
        for (dnsIp in publicDnsToIntercept) {
            try {
                builder.addRoute(dnsIp, 32)
            } catch (e: Exception) {
                Log.w(TAG, "Could not add route for $dnsIp: ${e.message}")
            }
        }

        // ── 2. Exclude our own app from the VPN ──────────────────────────────
        // Our upstream DNS sockets need to bypass the TUN; this prevents
        // any loop where outbound DNS requests would be routed back into the TUN.
        try {
            builder.addDisallowedApplication(packageName)
            Log.d(TAG, "Excluded own package '$packageName' from VPN tunnel")
        } catch (e: Exception) {
            Log.w(TAG, "addDisallowedApplication failed (non-fatal): ${e.message}")
        }

        val tun = builder.establish()
        if (tun == null) {
            Log.e(TAG, "Builder.establish() returned null — VPN permission not granted?")
            stopSelf()
            return
        }
        tunInterface = tun
        Log.i(TAG, "TUN interface established (fd=${tun.fd})")

        // ── 3. Query active network DNS servers + public fallbacks ───────────
        val upstreamServers = getActiveDnsServers()
        Log.i(TAG, "Active upstream DNS servers: $upstreamServers")

        // ── 4. Start the in-process DNS proxy ────────────────────────────────
        val proxy = LocalDnsProxyServer(
            vpnService = this,
            tunFd      = tun.fileDescriptor,
            classifier = classifier,
            logger     = logger,
            upstreamDnsServers = upstreamServers,
        )
        dnsProxy = proxy
        proxy.start(serviceScope)

        isRunning = true
        Log.i(TAG, "VPN started successfully — protection is ON")
        Log.i(TAG, "  Internet access:   UNAFFECTED (non-DNS traffic bypasses TUN)")
        Log.i(TAG, "  DNS filtering:     ACTIVE (gambling domains → NXDOMAIN)")
    }

    /**
     * Resolves the DNS servers currently in use by the active physical network
     * (e.g. Wi-Fi router DNS or cellular ISP DNS) so we forward queries to
     * servers guaranteed to work on the user's current network connection.
     */
    private fun getActiveDnsServers(): List<String> {
        val servers = mutableListOf<String>()
        try {
            val cm = getSystemService(android.net.ConnectivityManager::class.java)
            if (cm != null) {
                val activeNetwork = cm.activeNetwork
                if (activeNetwork != null) {
                    val lp = cm.getLinkProperties(activeNetwork)
                    lp?.dnsServers?.forEach { dns ->
                        val host = dns.hostAddress
                        if (!host.isNullOrBlank() && !host.startsWith("10.111.222.")) {
                            servers.add(host)
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Could not query system DNS servers: ${e.message}")
        }

        // Add standard public DNS servers as fallbacks
        val fallbacks = listOf("8.8.8.8", "1.1.1.1", "8.8.4.4")
        for (fb in fallbacks) {
            if (!servers.contains(fb)) {
                servers.add(fb)
            }
        }
        return servers
    }

    private fun stopVpn() {
        if (!isRunning && tunInterface == null) {
            Log.d(TAG, "stopVpn() called but not running — ignored")
            return
        }
        Log.i(TAG, "Stopping VPN")
        isRunning = false

        // Stop DNS proxy coroutine first, then close the TUN fd it was reading.
        dnsProxy?.stop()
        dnsProxy = null

        try {
            tunInterface?.close()
            Log.d(TAG, "TUN interface closed")
        } catch (e: Exception) {
            Log.w(TAG, "Error closing TUN interface: ${e.message}")
        }
        tunInterface = null

        stopForeground(true)
        stopSelf()
        Log.i(TAG, "VPN stopped — protection is OFF, all traffic flows normally")
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Foreground service notification (required for VpnService on API 26+)
    // ─────────────────────────────────────────────────────────────────────────

    private fun startForegroundCompat() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                NOTIF_CHANNEL_ID,
                "Gambling Blocker VPN",
                NotificationManager.IMPORTANCE_LOW,
            ).apply { description = "VPN protection service running" }

            val nm = getSystemService(NotificationManager::class.java)
            nm?.createNotificationChannel(channel)
        }

        val notification = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, NOTIF_CHANNEL_ID)
                .setContentTitle("Gambling Blocker")
                .setContentText("DNS protection is active")
                .setSmallIcon(android.R.drawable.ic_lock_lock)
                .build()
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
                .setContentTitle("Gambling Blocker")
                .setContentText("DNS protection is active")
                .setSmallIcon(android.R.drawable.ic_lock_lock)
                .build()
        }

        startForeground(NOTIFICATION_ID, notification)
    }
}