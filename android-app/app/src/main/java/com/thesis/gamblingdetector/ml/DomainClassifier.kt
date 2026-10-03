package com.thesis.gamblingdetector.ml

import android.content.Context
import android.util.Log
import org.json.JSONObject
import kotlin.math.exp

/**
 * Loads the folded logistic-regression weights exported by
 * ml-training/train_model.py (assets/model_weights.json) and runs
 * inference entirely on-device with no external ML runtime.
 *
 * Provides:
 *  - URL / Domain normalization (strips http/https, ports, paths, query params, www)
 *  - Dual evaluation (full hostname and registered apex domain)
 *  - Safe-list whitelist guarantee (0% FP for critical services)
 *  - Hybrid heuristic booster for stealth gambling compounds
 *  - Fail-secure fallback if weights fail to load
 *  - Detailed diagnostic logging
 */
class DomainClassifier(context: Context) {

    data class Prediction(
        val domain: String,
        val score: Double,
        val isGambling: Boolean,
        val features: DomainFeatures? = null,
        val note: String = "",
    )

    companion object {
        private const val TAG = "DomainClassifier"

        /**
         * Curated safe-list of prominent legitimate apex domains and public services.
         * Queries matching these or their subdomains are always allowed (0% False Positive guarantee).
         */
        val SAFE_DOMAINS = setOf(
            "google.com", "gstatic.com", "googleapis.com", "googleusercontent.com",
            "youtube.com", "ytimg.com", "googlevideo.com",
            "wikipedia.org", "wikimedia.org", "github.com", "githubusercontent.com",
            "microsoft.com", "apple.com", "amazon.com", "cloudflare.com",
            "linkedin.com", "windows.com", "live.com", "office.com",
            "mandiri.co.id", "bankmandiri.co.id", "bca.co.id", "klikbca.com",
            "bri.co.id", "bni.co.id", "cimbniaga.co.id", "dana.id", "ovo.id",
            "kemdikbud.go.id", "kominfo.go.id", "go.id", "ac.id",
            "detik.com", "kompas.com", "tempo.co", "tribunnews.com",
            "tokopedia.com", "shopee.co.id", "bukalapak.com", "gojek.com", "grab.com",
            "whatsapp.com", "whatsapp.net", "instagram.com", "facebook.com", "fbcdn.net",
            "twitter.com", "x.com", "tiktok.com", "byteoversea.com",
        )

        /**
         * Robust normalizer: strips scheme (http/https), credentials, ports, paths,
         * queries, fragments, and trailing slashes/dots from any input URL or domain.
         */
        fun normalize(rawInput: String): String {
            var s = rawInput.trim().lowercase()
            if (s.isEmpty()) return ""

            // Strip scheme (http://, https://, ftp://, etc.)
            val schemeIdx = s.indexOf("://")
            if (schemeIdx != -1) {
                s = s.substring(schemeIdx + 3)
            }

            // Strip userinfo (user:pass@host)
            val atIdx = s.indexOf('@')
            if (atIdx != -1) {
                s = s.substring(atIdx + 1)
            }

            // Cut off path, query string, and fragment
            for (delim in charArrayOf('/', '?', '#')) {
                val idx = s.indexOf(delim)
                if (idx != -1) {
                    s = s.substring(0, idx)
                }
            }

            // Cut off port (:80, :443, :8080)
            val colonIdx = s.lastIndexOf(':')
            if (colonIdx != -1 && !s.contains(']')) { // Avoid IPv6 brackets
                val portCandidate = s.substring(colonIdx + 1)
                if (portCandidate.all { it.isDigit() }) {
                    s = s.substring(0, colonIdx)
                }
            }

            return s.trim('.')
        }

        /**
         * Extracts the apex domain if possible (e.g. www.gbwinlink.org -> gbwinlink.org).
         */
        fun getApexDomain(domain: String): String {
            val parts = domain.split('.')
            if (parts.size <= 2) return domain
            // If domain starts with common subdomain prefix (www, m, play, login, app)
            if (parts[0] in setOf("www", "m", "mobile", "app", "play", "login", "link", "web")) {
                return parts.drop(1).joinToString(".")
            }
            // Generic SLD + TLD for multi-label domains (e.g. sub.example.co.id)
            if (parts.size >= 3 && parts[parts.size - 2] in setOf("co", "go", "ac", "or", "net", "com") && parts.last().length == 2) {
                return parts.takeLast(3).joinToString(".")
            }
            return parts.takeLast(2).joinToString(".")
        }
    }

    var isModelLoaded: Boolean = false
        private set
    var modelLoadError: String? = null
        private set

    private var weights: DoubleArray = DoubleArray(0)
    private var bias: Double = 0.0
    private var threshold: Double = 0.5

    init {
        try {
            val json = context.assets.open("model_weights.json")
                .bufferedReader()
                .use { it.readText() }
            val obj = JSONObject(json)
            val weightsArray = obj.getJSONArray("weights")
            weights = DoubleArray(weightsArray.length()) { i -> weightsArray.getDouble(i) }
            bias = obj.getDouble("bias")
            threshold = obj.optDouble("threshold", 0.5)
            isModelLoaded = true
            Log.i(TAG, "ML Model loaded successfully: ${weights.size} features, bias=$bias, threshold=$threshold")
        } catch (e: Exception) {
            isModelLoaded = false
            modelLoadError = e.message
            Log.e(TAG, "FATAL: Failed to load ML model weights: ${e.message}", e)
        }
    }

    private fun sigmoid(z: Double): Double = 1.0 / (1.0 + exp(-z))

    /**
     * Evaluates a single clean domain name using Logistic Regression weights.
     */
    private fun predictScore(cleanDomain: String, featureObj: DomainFeatures): Double {
        if (!isModelLoaded || weights.isEmpty()) {
            return fallbackHeuristicScore(cleanDomain, featureObj)
        }

        val features = featureObj.toVector()
        require(features.size == weights.size) {
            "Feature vector size (${features.size}) does not match model weights size (${weights.size})."
        }

        var z = bias
        for (i in features.indices) {
            z += features[i] * weights[i]
        }
        var score = sigmoid(z)

        // Zero-indicator safeguard:
        // A domain with ZERO gambling-specific signals (no gambling keywords, no lucky numbers,
        // no gambling compounds, no suspicious TLD, not a direct IP, no compromised pattern)
        // should never be falsely classified as gambling due to incidental digits or length.
        val hasAnyIndicator = featureObj.gamblingKeywordHits > 0.0 ||
            featureObj.hasLuckyNumber == 1.0 ||
            featureObj.hasGamblingCompound == 1.0 ||
            featureObj.suspiciousTld == 1.0 ||
            featureObj.isIpAddress == 1.0 ||
            featureObj.hasCompromisedPattern == 1.0

        if (!hasAnyIndicator) {
            score = minOf(score, 0.35)
        }

        // Hybrid Heuristic Booster:
        // Ensures strong compounds (winlink, slot+lucky_number, direct IP gambling host, etc.)
        // are always classified firmly
        if (featureObj.hasGamblingCompound == 1.0 ||
            (featureObj.gamblingKeywordHits >= 1.0 && featureObj.hasLuckyNumber == 1.0) ||
            featureObj.isIpAddress == 1.0) {
            score = maxOf(score, 0.95)
        }

        return score
    }

    /**
     * Fail-secure fallback when model weights cannot be loaded:
     * Evaluates domain using lexical rule heuristics so threats are NOT silently allowed.
     */
    private fun fallbackHeuristicScore(cleanDomain: String, featureObj: DomainFeatures): Double {
        Log.w(TAG, "Using FAIL-SECURE fallback heuristic for domain='$cleanDomain' (model unready)")
        var score = 0.05
        if (featureObj.hasGamblingCompound == 1.0) score += 0.85
        if (featureObj.isIpAddress == 1.0) score += 0.85
        if (featureObj.gamblingKeywordHits > 0) score += (featureObj.gamblingKeywordHits * 0.3)
        if (featureObj.hasLuckyNumber == 1.0) score += 0.3
        if (featureObj.suspiciousTld == 1.0) score += 0.2
        return minOf(1.0, score)
    }

    /**
     * Classifies [rawInput] (which can be a full URL, hostname, or domain query).
     * Applies identical preprocessing, feature extraction, and dual evaluation.
     */
    fun classify(rawInput: String): Prediction {
        val cleanDomain = normalize(rawInput)
        if (cleanDomain.isEmpty()) {
            return Prediction(domain = "", score = 0.0, isGambling = false, note = "Empty input")
        }

        // 1. Safe-list check: guarantee no false positives for critical services
        for (safe in SAFE_DOMAINS) {
            if (cleanDomain == safe || cleanDomain.endsWith(".$safe")) {
                Log.d(TAG, "[SAFE-LIST ALLOW] domain='$cleanDomain' matches whitelist '$safe'")
                return Prediction(domain = cleanDomain, score = 0.0, isGambling = false, note = "Safe-listed")
            }
        }

        // 2. Primary evaluation on normalized domain
        val featureObj = DomainFeatureExtractor.extract(cleanDomain)
        var primaryScore = predictScore(cleanDomain, featureObj)

        // 3. Dual evaluation on apex domain if subdomain exists (e.g. www.gbwinlink.org -> gbwinlink.org)
        val apexDomain = getApexDomain(cleanDomain)
        var evaluatedFeatures = featureObj
        if (apexDomain != cleanDomain && apexDomain.isNotEmpty()) {
            // Also check apex against safe-list
            if (apexDomain in SAFE_DOMAINS) {
                return Prediction(domain = cleanDomain, score = 0.0, isGambling = false, note = "Apex safe-listed")
            }

            val apexFeatures = DomainFeatureExtractor.extract(apexDomain)
            val apexScore = predictScore(apexDomain, apexFeatures)
            if (apexScore > primaryScore) {
                primaryScore = apexScore
                evaluatedFeatures = apexFeatures
            }
        }

        val isGambling = primaryScore >= threshold
        val action = if (isGambling) "BLOCKED" else "ALLOWED"

        Log.i(
            TAG,
            "[ML PREDICTION] domain='$cleanDomain' (raw='$rawInput') -> " +
                "score=${"%.4f".format(primaryScore)} threshold=$threshold action=$action " +
                "features=[kw=${evaluatedFeatures.gamblingKeywordHits.toInt()}, " +
                "compound=${evaluatedFeatures.hasGamblingCompound.toInt()}, " +
                "lucky=${evaluatedFeatures.hasLuckyNumber.toInt()}, " +
                "density=${"%.2f".format(evaluatedFeatures.keywordDensity)}, " +
                "ip=${evaluatedFeatures.isIpAddress.toInt()}, " +
                "compromised=${evaluatedFeatures.hasCompromisedPattern.toInt()}]"
        )

        return Prediction(
            domain = cleanDomain,
            score = primaryScore,
            isGambling = isGambling,
            features = evaluatedFeatures,
            note = if (isGambling) "Exceeds threat threshold ($threshold)" else "Below threat threshold",
        )
    }
}
