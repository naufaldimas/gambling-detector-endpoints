package com.thesis.gamblingdetector.data

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

data class DetectionEvent(
    val domain: String,
    val action: String, // "BLOCKED" or "ALLOWED"
    val score: Double,
    val timestamp: Long,
)

/**
 * Lightweight local-only logging of detection events. Useful for:
 *  - The in-app UI (recent blocks / activity feed)
 *  - Exporting logs for your thesis evaluation (precision/recall against
 *    a ground-truth labeled test set, false positive rate, latency, etc.)
 *
 * All data stays on-device — no telemetry is sent anywhere, consistent
 * with the on-device/privacy-friendly design decision.
 */
class DetectionLogger(context: Context) : SQLiteOpenHelper(context, DB_NAME, null, DB_VERSION) {

    companion object {
        private const val DB_NAME = "detection_log.db"
        private const val DB_VERSION = 1
        private const val TABLE = "detections"
    }

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            """
            CREATE TABLE $TABLE (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                action TEXT NOT NULL,
                score REAL NOT NULL,
                timestamp INTEGER NOT NULL
            )
            """.trimIndent()
        )
        db.execSQL("CREATE INDEX idx_timestamp ON $TABLE(timestamp DESC)")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        db.execSQL("DROP TABLE IF EXISTS $TABLE")
        onCreate(db)
    }

    fun logDetection(domain: String, action: String, score: Double) {
        val values = ContentValues().apply {
            put("domain", domain)
            put("action", action)
            put("score", score)
            put("timestamp", System.currentTimeMillis())
        }
        writableDatabase.insert(TABLE, null, values)
    }

    fun getRecent(limit: Int = 100): List<DetectionEvent> {
        val events = mutableListOf<DetectionEvent>()
        readableDatabase.rawQuery(
            "SELECT domain, action, score, timestamp FROM $TABLE ORDER BY timestamp DESC LIMIT ?",
            arrayOf(limit.toString())
        ).use { cursor ->
            while (cursor.moveToNext()) {
                events.add(
                    DetectionEvent(
                        domain = cursor.getString(0),
                        action = cursor.getString(1),
                        score = cursor.getDouble(2),
                        timestamp = cursor.getLong(3),
                    )
                )
            }
        }
        return events
    }

    fun getStats(): Pair<Int, Int> {
        // returns (blockedCount, allowedCount) — handy for a summary dashboard
        var blocked = 0
        var allowed = 0
        readableDatabase.rawQuery(
            "SELECT action, COUNT(*) FROM $TABLE GROUP BY action", null
        ).use { cursor ->
            while (cursor.moveToNext()) {
                val action = cursor.getString(0)
                val count = cursor.getInt(1)
                if (action == "BLOCKED") blocked = count else allowed = count
            }
        }
        return blocked to allowed
    }

    fun clearLogs() {
        writableDatabase.delete(TABLE, null, null)
    }
}
