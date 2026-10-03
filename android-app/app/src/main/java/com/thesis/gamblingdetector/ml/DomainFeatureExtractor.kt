package com.thesis.gamblingdetector.ml

import kotlin.math.ln

/**
 * Domain Feature Extraction
 * =========================
 *
 * MUST stay in exact sync with ml-training/features.py (extract_features_extended).
 *
 * Feature vector order (must match model_weights.json "feature_names", 17 features):
 *   0. domain_length
 *   1. subdomain_count
 *   2. digit_ratio
 *   3. hyphen_count
 *   4. vowel_ratio
 *   5. consonant_run_max
 *   6. shannon_entropy
 *   7. gambling_keyword_hits
 *   8. suspicious_tld
 *   9. digit_letter_transition
 *   10. length_of_sld
 *   11. has_random_looking_prefix
 *   12. has_lucky_number
 *   13. has_gambling_compound
 *   14. keyword_density
 *   15. is_ip_address
 *   16. has_compromised_pattern
 */

data class DomainFeatures(
    val domainLength: Double,
    val subdomainCount: Double,
    val digitRatio: Double,
    val hyphenCount: Double,
    val vowelRatio: Double,
    val consonantRunMax: Double,
    val shannonEntropy: Double,
    val gamblingKeywordHits: Double,
    val suspiciousTld: Double,
    val digitLetterTransition: Double,
    val lengthOfSld: Double,
    val hasRandomLookingPrefix: Double,
    val hasLuckyNumber: Double,
    val hasGamblingCompound: Double,
    val keywordDensity: Double,
    val isIpAddress: Double,
    val hasCompromisedPattern: Double,
) {
    fun toVector(): DoubleArray = doubleArrayOf(
        domainLength, subdomainCount, digitRatio, hyphenCount, vowelRatio,
        consonantRunMax, shannonEntropy, gamblingKeywordHits, suspiciousTld,
        digitLetterTransition, lengthOfSld, hasRandomLookingPrefix,
        hasLuckyNumber, hasGamblingCompound, keywordDensity,
        isIpAddress, hasCompromisedPattern,
    )
}

object DomainFeatureExtractor {

    // Keep in exact sync with GAMBLING_KEYWORDS in ml-training/features.py
    private val GAMBLING_KEYWORDS = listOf(
        "slot", "gacor", "togel", "toto", "judi", "judol", "bet", "casino", "poker",
        "domino", "qq", "sbobet", "maxwin", "jackpot", "gaple", "rtp",
        "spin", "gambling", "wager", "roulette", "baccarat", "pragmatic",
        "hoki", "menang", "jitu", "prediksi", "taruhan", "bandar", "agen",
        "parlay", "livecasino", "sensasional", "cuan", "rungkat", "rungkad",
        "habanero", "pgsoft", "spadegaming", "zeus", "olympus", "mahjong",
        "bola", "angka", "depo", "wd",
    )

    // Lucky numbers heavily used as branding by gambling sites (e.g. slot88, cuan777, 303)
    private val LUCKY_NUMBERS = listOf(
        "88", "77", "99", "777", "888", "999", "138", "188",
        "303", "5000", "10k", "25k", "8888", "7777",
    )

    private val GAMBLING_COMPOUNDS = listOf(
        "winlink", "linkwin", "linkalt", "alternatif", "linkdaftar", "linklogin",
        "linkgacor", "linkslot", "linkresmi", "megawin", "bigwin", "cuanwin",
        "superwin", "judibola", "depopulsa", "depodana", "agenslot", "bandartogel",
        "slotresmi", "bocoran", "anti-rungkat", "antirungkad", "slotonline",
        "bocorangacor", "rajawin", "kingwin", "hokiterus", "gudangcuan",
    )

    // Keep in sync with SUSPICIOUS_TLDS in ml-training/features.py
    private val SUSPICIOUS_TLDS = setOf(
        "xyz", "top", "club", "online", "site", "info", "vip", "biz",
        "live", "fun", "click", "link", "icu", "cc", "me", "pro",
        "asia", "space", "bet", "rest", "buzz", "monster",
    )

    private val VOWELS = setOf('a', 'e', 'i', 'o', 'u')

    /** IPv4 regex: e.g. 192.168.1.1 or 117.18.0.168 */
    private val IPV4_REGEX = Regex("""^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$""")

    /**
     * Compromised/injection patterns (keep in sync with features.py COMPROMISED_PATTERNS):
     *  - alphanum run like slot2024win
     *  - keyword + action injection like slot-login, judi-daftar, rtp-live
     *  - compromised Indonesian association domains (pafi*)
     */
    private val COMPROMISED_REGEX = Regex(
        """[a-z]{2,4}\d{4,}[a-z]{2,4}""" +
        """|(slot|togel|judi|casino|bet|gacor|poker|cuan|maxwin)[-_]?(login|daftar|masuk|link|apk|rtp)""" +
        """|pafi[a-z0-9]+"""
    )

    private fun isIpv4(s: String): Boolean {
        val m = IPV4_REGEX.matchEntire(s) ?: return false
        return m.groupValues.drop(1).all { g -> g.toIntOrNull()?.let { it in 0..255 } == true }
    }

    private fun shannonEntropy(s: String): Double {
        if (s.isEmpty()) return 0.0
        val freq = HashMap<Char, Int>()
        for (ch in s) freq[ch] = (freq[ch] ?: 0) + 1
        val n = s.length.toDouble()
        return -freq.values.sumOf { c ->
            val p = c / n
            p * (ln(p) / ln(2.0))
        }
    }

    private fun longestConsonantRun(s: String): Int {
        var maxRun = 0
        var cur = 0
        for (ch in s.lowercase()) {
            if (ch.isLetter() && ch !in VOWELS) {
                cur += 1
                if (cur > maxRun) maxRun = cur
            } else {
                cur = 0
            }
        }
        return maxRun
    }

    private fun digitLetterTransitions(s: String): Int {
        var count = 0
        var prevType: Char? = null // 'd' digit, 'l' letter, 'o' other
        for (ch in s) {
            val curType = when {
                ch.isDigit() -> 'd'
                ch.isLetter() -> 'l'
                else -> 'o'
            }
            if (prevType != null && curType != prevType && curType != 'o' && prevType != 'o') {
                count += 1
            }
            prevType = curType
        }
        return count
    }

    private fun splitDomain(domain: String): Triple<List<String>, String, String> {
        val clean = domain.lowercase().trim('.')
        val labels = clean.split(".")
        if (labels.size < 2) {
            return Triple(labels, labels.getOrElse(0) { "" }, "")
        }
        val tld = labels.last()
        val sld = labels[labels.size - 2]
        return Triple(labels, sld, tld)
    }

    fun extract(domainInput: String): DomainFeatures {
        val domain = domainInput.lowercase().trim()
        val (labels, sld, tld) = splitDomain(domain)

        val length = domain.length
        val subdomainCount = maxOf(0, labels.size - 2)
        val digits = domain.count { it.isDigit() }
        val digitRatio = if (length > 0) digits.toDouble() / length else 0.0
        val hyphenCount = domain.count { it == '-' }
        val alphaChars = domain.filter { it.isLetter() }
        val vowelCount = alphaChars.count { it in VOWELS }
        val vowelRatio = if (alphaChars.isNotEmpty()) vowelCount.toDouble() / alphaChars.length else 0.0
        val consonantRunMax = longestConsonantRun(domain)
        val entropy = shannonEntropy(domain)

        // Keyword hits in SLD
        var matchedKeywordLen = 0
        var keywordHits = 0
        for (kw in GAMBLING_KEYWORDS) {
            if (sld.contains(kw)) {
                keywordHits += 1
                matchedKeywordLen += kw.length
            }
        }

        // Lucky number detection
        val hasLuckyNum = if (LUCKY_NUMBERS.any { sld.contains(it) }) 1 else 0

        // Compound gambling detection
        var hasCompound = 0
        for (compound in GAMBLING_COMPOUNDS) {
            if (sld.contains(compound)) {
                hasCompound = 1
                matchedKeywordLen += compound.length
                break
            }
        }

        // Also detect win+link or win+lucky_number pattern
        if (sld.contains("win") && (sld.contains("link") || hasLuckyNum == 1)) {
            hasCompound = 1
            matchedKeywordLen += 6
        }

        val sldLen = sld.length
        val keywordDensity = if (sldLen > 0) minOf(1.0, matchedKeywordLen.toDouble() / sldLen) else 0.0

        val suspiciousTld = if (tld in SUSPICIOUS_TLDS || domain.endsWith(".pages.dev") || domain.endsWith(".workers.dev")) 1 else 0
        val transitions = digitLetterTransitions(sld)
        val randomPrefix = if (sldLen in 3..8 && entropy >= 2.5 && digits > 0) 1 else 0

        // Feature 15: is_ip_address (raw IPv4 address)
        val isIp = if (isIpv4(domain)) 1 else 0

        // Feature 16: has_compromised_pattern
        val hasCompromised = if (COMPROMISED_REGEX.containsMatchIn(domain)) 1 else 0

        return DomainFeatures(
            domainLength = length.toDouble(),
            subdomainCount = subdomainCount.toDouble(),
            digitRatio = digitRatio,
            hyphenCount = hyphenCount.toDouble(),
            vowelRatio = vowelRatio,
            consonantRunMax = consonantRunMax.toDouble(),
            shannonEntropy = entropy,
            gamblingKeywordHits = keywordHits.toDouble(),
            suspiciousTld = suspiciousTld.toDouble(),
            digitLetterTransition = transitions.toDouble(),
            lengthOfSld = sldLen.toDouble(),
            hasRandomLookingPrefix = randomPrefix.toDouble(),
            hasLuckyNumber = hasLuckyNum.toDouble(),
            hasGamblingCompound = hasCompound.toDouble(),
            keywordDensity = keywordDensity,
            isIpAddress = isIp.toDouble(),
            hasCompromisedPattern = hasCompromised.toDouble(),
        )
    }
}
