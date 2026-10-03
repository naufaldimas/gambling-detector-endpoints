package com.thesis.gamblingdetector.ui

import android.content.Context
import android.content.Intent
import android.net.VpnService
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.EditText
import android.widget.ImageView
import android.widget.ListView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.chip.ChipGroup
import com.thesis.gamblingdetector.R
import com.thesis.gamblingdetector.data.DetectionEvent
import com.thesis.gamblingdetector.data.DetectionLogger
import com.thesis.gamblingdetector.ml.DomainClassifier
import com.thesis.gamblingdetector.vpn.GamblingVpnService
import java.text.SimpleDateFormat
import java.util.Locale

/**
 * Modern, interactive dashboard for Gambling Detector:
 *  - Real-time VPN protection status & hero card
 *  - KPI metrics (total queries, threats blocked, safe connections)
 *  - Interactive on-device ML domain classifier sandbox / tester
 *  - Filterable live activity logs with custom rich list items
 */
class MainActivity : AppCompatActivity() {

    private lateinit var logger: DetectionLogger
    private lateinit var classifier: DomainClassifier

    private lateinit var heroCard: MaterialCardView
    private lateinit var statusDot: View
    private lateinit var statusTitle: TextView
    private lateinit var statusSubtitle: TextView
    private lateinit var toggleButton: MaterialButton

    private lateinit var badgeMlEngineStatus: TextView
    private lateinit var textLastInspected: TextView
    private lateinit var textLastActionBadge: TextView

    private lateinit var statTotal: TextView
    private lateinit var statBlocked: TextView
    private lateinit var statAllowed: TextView

    private lateinit var inputTestDomain: EditText
    private lateinit var btnTestDomain: MaterialButton
    private lateinit var layoutTestResult: View
    private lateinit var textTestResultBadge: TextView
    private lateinit var textTestResultDetails: TextView
    private lateinit var textTestResultSubDetails: TextView
    
    private lateinit var inputDashboardIp: EditText
    private lateinit var btnSaveIp: MaterialButton

    private lateinit var filterChipGroup: ChipGroup
    private lateinit var emptyStateText: TextView
    private lateinit var logListView: ListView

    private var isProtectionOn = false
    private var currentFilter: String? = null // null = All, "BLOCKED", "ALLOWED"

    private val vpnPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            startVpnService()
        } else {
            Toast.makeText(this, "VPN permission denied — protection cannot start.", Toast.LENGTH_LONG).show()
            updateProtectionUi(false)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        logger = DetectionLogger(applicationContext)
        classifier = DomainClassifier(applicationContext)

        initViews()
        setupListeners()
        updateProtectionUi(false)
        refreshLog()
    }

    override fun onResume() {
        super.onResume()
        refreshLog()
    }

    private fun initViews() {
        badgeMlEngineStatus = findViewById(R.id.badgeMlEngineStatus)
        if (classifier.isModelLoaded) {
            badgeMlEngineStatus.text = "ML Engine Ready (15 Feat)"
            badgeMlEngineStatus.background = ContextCompat.getDrawable(this, R.drawable.bg_badge_ml_ready)
            badgeMlEngineStatus.setTextColor(ContextCompat.getColor(this, R.color.ml_ready_text))
        } else {
            badgeMlEngineStatus.text = "ML Engine Fallback"
            badgeMlEngineStatus.background = ContextCompat.getDrawable(this, R.drawable.bg_badge_ml_error)
            badgeMlEngineStatus.setTextColor(ContextCompat.getColor(this, R.color.ml_error_text))
        }

        heroCard = findViewById(R.id.heroCard)
        statusDot = findViewById(R.id.statusDot)
        statusTitle = findViewById(R.id.statusTitle)
        statusSubtitle = findViewById(R.id.statusSubtitle)
        toggleButton = findViewById(R.id.toggleButton)

        textLastInspected = findViewById(R.id.textLastInspected)
        textLastActionBadge = findViewById(R.id.textLastActionBadge)

        statTotal = findViewById(R.id.statTotal)
        statBlocked = findViewById(R.id.statBlocked)
        statAllowed = findViewById(R.id.statAllowed)

        inputTestDomain = findViewById(R.id.inputTestDomain)
        btnTestDomain = findViewById(R.id.btnTestDomain)
        layoutTestResult = findViewById(R.id.layoutTestResult)
        textTestResultBadge = findViewById(R.id.textTestResultBadge)
        textTestResultDetails = findViewById(R.id.textTestResultDetails)
        textTestResultSubDetails = findViewById(R.id.textTestResultSubDetails)
        
        inputDashboardIp = findViewById(R.id.inputDashboardIp)
        btnSaveIp = findViewById(R.id.btnSaveIp)
        
        val prefs = getSharedPreferences("AppPrefs", Context.MODE_PRIVATE)
        val savedUrl = prefs.getString("dashboard_url", "")
        if (!savedUrl.isNullOrEmpty()) {
            inputDashboardIp.setText(savedUrl)
        }

        filterChipGroup = findViewById(R.id.filterChipGroup)
        emptyStateText = findViewById(R.id.emptyStateText)
        logListView = findViewById(R.id.logListView)
    }

    private fun setupListeners() {
        // Toggle Protection Button
        toggleButton.setOnClickListener {
            if (!isProtectionOn) {
                requestVpnPermissionAndStart()
            } else {
                stopVpnService()
            }
        }
        
        // Save Dashboard IP Config
        btnSaveIp.setOnClickListener {
            var url = inputDashboardIp.text?.toString()?.trim() ?: ""
            if (url.isNotEmpty() && !url.startsWith("http")) {
                url = "http://$url"
            }
            if (url.endsWith("/")) {
                url = url.substring(0, url.length - 1)
            }
            
            val prefs = getSharedPreferences("AppPrefs", Context.MODE_PRIVATE)
            prefs.edit().putString("dashboard_url", url).apply()
            
            inputDashboardIp.setText(url)
            Toast.makeText(this, "Dashboard URL saved", Toast.LENGTH_SHORT).show()
        }

        // Live Domain / URL Tester Button
        btnTestDomain.setOnClickListener {
            val rawInput = inputTestDomain.text?.toString()?.trim() ?: ""
            if (rawInput.isEmpty()) {
                Toast.makeText(this, "Please enter a URL or domain to test", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val prediction = classifier.classify(rawInput)
            layoutTestResult.visibility = View.VISIBLE
            val percent = "%.1f".format(prediction.score * 100)

            if (prediction.isGambling) {
                textTestResultBadge.text = "BLOCKED"
                textTestResultBadge.setTextColor(ContextCompat.getColor(this, R.color.blocked_red))
                textTestResultBadge.background = ContextCompat.getDrawable(this, R.drawable.bg_badge_blocked)
                textTestResultDetails.text = "Identified as Gambling ($percent% threat score)"
            } else {
                textTestResultBadge.text = "ALLOWED"
                textTestResultBadge.setTextColor(ContextCompat.getColor(this, R.color.allowed_green))
                textTestResultBadge.background = ContextCompat.getDrawable(this, R.drawable.bg_badge_allowed)
                textTestResultDetails.text = "Legitimate Domain ($percent% threat score)"
            }

            val feats = prediction.features
            val kwHits = feats?.gamblingKeywordHits?.toInt() ?: 0
            val hasCompound = if (feats?.hasGamblingCompound == 1.0) "Yes" else "No"
            val hasLucky = if (feats?.hasLuckyNumber == 1.0) "Yes" else "No"
            val densityStr = "%.0f".format((feats?.keywordDensity ?: 0.0) * 100)
            textTestResultSubDetails.text = "Domain: ${prediction.domain} • Keywords: $kwHits • Compound: $hasCompound • Lucky Num: $hasLucky • Density: $densityStr%"

            // Log the test in real-time activity log as well
            logger.logDetection(prediction.domain, if (prediction.isGambling) "BLOCKED" else "ALLOWED", prediction.score)
            refreshLog()
        }

        // Refresh Log Button
        findViewById<View>(R.id.btnRefreshLog).setOnClickListener {
            refreshLog()
            Toast.makeText(this, "Log refreshed", Toast.LENGTH_SHORT).show()
        }

        // Clear Log Button
        findViewById<View>(R.id.btnClearLog).setOnClickListener {
            AlertDialog.Builder(this)
                .setTitle("Clear Activity Log?")
                .setMessage("Are you sure you want to delete all recorded detection events?")
                .setPositiveButton("Clear") { _, _ ->
                    logger.clearLogs()
                    refreshLog()
                    Toast.makeText(this, "Logs cleared", Toast.LENGTH_SHORT).show()
                }
                .setNegativeButton("Cancel", null)
                .show()
        }

        // Filter Chips
        filterChipGroup.setOnCheckedStateChangeListener { _, checkedIds ->
            currentFilter = when {
                checkedIds.contains(R.id.chipBlocked) -> "BLOCKED"
                checkedIds.contains(R.id.chipAllowed) -> "ALLOWED"
                else -> null
            }
            refreshLog()
        }
    }

    private fun updateProtectionUi(active: Boolean) {
        isProtectionOn = active
        if (active) {
            statusDot.background = ContextCompat.getDrawable(this, R.drawable.bg_status_dot_on)
            statusTitle.text = "Connected"
            statusTitle.setTextColor(ContextCompat.getColor(this, R.color.status_active))
            statusSubtitle.text = "You are protected."

            toggleButton.text = "TAP TO\nDISCONNECT"
            toggleButton.setBackgroundColor(ContextCompat.getColor(this, R.color.status_active_border))
            toggleButton.icon = null
        } else {
            statusDot.background = ContextCompat.getDrawable(this, R.drawable.bg_status_dot_off)
            statusTitle.text = "Not Connected"
            statusTitle.setTextColor(ContextCompat.getColor(this, R.color.text_primary))
            statusSubtitle.text = "Your network is unprotected."

            toggleButton.text = "TAP TO\nCONNECT"
            toggleButton.setBackgroundColor(ContextCompat.getColor(this, R.color.status_inactive_border))
            toggleButton.icon = null
        }
    }

    private fun requestVpnPermissionAndStart() {
        val intent = VpnService.prepare(this)
        if (intent != null) {
            vpnPermissionLauncher.launch(intent)
        } else {
            startVpnService()
        }
    }

    private fun startVpnService() {
        val intent = Intent(this, GamblingVpnService::class.java).apply {
            action = GamblingVpnService.ACTION_START
        }
        startService(intent)
        updateProtectionUi(true)
    }

    private fun stopVpnService() {
        val intent = Intent(this, GamblingVpnService::class.java).apply {
            action = GamblingVpnService.ACTION_STOP
        }
        startService(intent)
        updateProtectionUi(false)
    }

    private fun refreshLog() {
        val (blocked, allowed) = logger.getStats()
        statTotal.text = (blocked + allowed).toString()
        statBlocked.text = blocked.toString()
        statAllowed.text = allowed.toString()

        val allRecent = logger.getRecent(1)
        if (allRecent.isNotEmpty()) {
            val latest = allRecent.first()
            val percent = (latest.score * 100).toInt()
            textLastInspected.text = "${latest.domain} • ${percent}% threat"
            textLastActionBadge.text = latest.action
            if (latest.action == "BLOCKED") {
                textLastActionBadge.setTextColor(ContextCompat.getColor(this, R.color.blocked_red))
                textLastActionBadge.background = ContextCompat.getDrawable(this, R.drawable.bg_badge_blocked)
            } else {
                textLastActionBadge.setTextColor(ContextCompat.getColor(this, R.color.allowed_green))
                textLastActionBadge.background = ContextCompat.getDrawable(this, R.drawable.bg_badge_allowed)
            }
        } else {
            textLastInspected.text = if (isProtectionOn) "Protection active • listening for DNS queries..." else "Protection is OFF"
            textLastActionBadge.text = if (isProtectionOn) "ACTIVE" else "STANDBY"
            textLastActionBadge.setTextColor(ContextCompat.getColor(this, R.color.text_muted))
            textLastActionBadge.background = ContextCompat.getDrawable(this, R.drawable.bg_card_secondary)
        }

        var events = logger.getRecent(100)
        if (currentFilter != null) {
            events = events.filter { it.action == currentFilter }
        }

        if (events.isEmpty()) {
            emptyStateText.visibility = View.VISIBLE
            logListView.visibility = View.GONE
        } else {
            emptyStateText.visibility = View.GONE
            logListView.visibility = View.VISIBLE
            logListView.adapter = DetectionLogAdapter(this, events)
        }
    }

    /** Custom adapter rendering styled card items for detection logs */
    private class DetectionLogAdapter(
        context: Context,
        private val events: List<DetectionEvent>
    ) : ArrayAdapter<DetectionEvent>(context, 0, events) {

        private val fmt = SimpleDateFormat("HH:mm:ss", Locale.getDefault())

        override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
            val view = convertView ?: LayoutInflater.from(context).inflate(
                R.layout.item_detection_log, parent, false
            )
            val event = events[position]

            val iconAction = view.findViewById<ImageView>(R.id.iconAction)
            val textDomain = view.findViewById<TextView>(R.id.textDomain)
            val textDetails = view.findViewById<TextView>(R.id.textDetails)
            val badgeAction = view.findViewById<TextView>(R.id.badgeAction)

            textDomain.text = event.domain
            val percent = (event.score * 100).toInt()
            val timeStr = fmt.format(event.timestamp)
            textDetails.text = "Threat Score: $percent% • $timeStr"

            if (event.action == "BLOCKED") {
                badgeAction.text = "BLOCKED"
                badgeAction.setTextColor(ContextCompat.getColor(context, R.color.blocked_red))
                badgeAction.background = ContextCompat.getDrawable(context, R.drawable.bg_badge_blocked)
                iconAction.setImageResource(R.drawable.ic_block)
                iconAction.background = ContextCompat.getDrawable(context, R.drawable.bg_badge_blocked)
            } else {
                badgeAction.text = "ALLOWED"
                badgeAction.setTextColor(ContextCompat.getColor(context, R.color.allowed_green))
                badgeAction.background = ContextCompat.getDrawable(context, R.drawable.bg_badge_allowed)
                iconAction.setImageResource(R.drawable.ic_check)
                iconAction.background = ContextCompat.getDrawable(context, R.drawable.bg_badge_allowed)
            }

            return view
        }
    }
}
