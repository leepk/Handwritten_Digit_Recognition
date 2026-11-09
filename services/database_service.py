import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent 
DB_NAME = BASE_DIR/"predictions.db"


def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreing_keys = ON")
    return conn


def create_table():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT,
                character TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def create_feedback_table():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL UNIQUE,
                actual_character TEXT NOT NULL,
                is_correct INTEGER NOT NULL,
                reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id)
                    REFERENCES predictions(id)
                    ON DELETE CASCADE
            )
        """)

def save_feedback(prediction_id, actual_character):
    with get_db_connection() as conn:
        prediction = conn.execute(
            "SELECT character FROM predictions WHERE id = ?",
            (prediction_id,)
        ).fetchone()

        if prediction is None:
            raise ValueError("Prediction not found")

        predicted_character = prediction["character"]
        is_correct = int(predicted_character == actual_character)

        conn.execute("""
            INSERT INTO prediction_feedback
                (prediction_id, actual_character, is_correct)
            VALUES (?, ?, ?)
            ON CONFLICT(prediction_id)
            DO UPDATE SET
                actual_character = excluded.actual_character,
                is_correct = excluded.is_correct,
                reviewed_at = CURRENT_TIMESTAMP
        """, (prediction_id, actual_character, is_correct))

def insert_prediction(image_path, character, confidence):
    with get_db_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO predictions (image_path, character, confidence)
            VALUES (?, ?, ?)
        """, (image_path, character, confidence))

        return cursor.lastrowid


def get_predictions():
    with get_db_connection() as conn:
        cursor = conn.execute("""
            SELECT *
            FROM predictions
            ORDER BY timestamp DESC
        """)
        return [dict(row) for row in cursor.fetchall()]

def search_predictions(
    start_date=None,
    end_date=None,
    min_confidence=0,
    max_confidence=100,
    character="All",
    character_type="All",
    sort_order="Newest first"
):
    query = """
        SELECT
            p.id,
            p.image_path,
            p.character,
            ROUND(confidence * 100, 2) AS confidence_percent,
            p.timestamp,
            f.actual_character,
            CASE
                WHEN f.is_correct = 1 THEN 'Correct'
                WHEN f.is_correct = 0 THEN 'Incorrect'
                ELSE 'Not Reviewed'
            END AS feedback_status
        FROM predictions p
        LEFT JOIN prediction_feedback f
            ON p.id = f.prediction_id
        WHERE confidence BETWEEN ? AND ?
    """

    params = [
        min_confidence / 100,
        max_confidence / 100
    ]

    if start_date is not None:
        query += " AND date(timestamp) >= ?"
        params.append(str(start_date))

    if end_date is not None:
        query += " AND date(timestamp) <= ?"
        params.append(str(end_date))

    if character != "All":
        query += " AND character = ? COLLATE BINARY"
        params.append(character)

    if character_type == "Digit":
        query += " AND character GLOB '[0-9]'"
    elif character_type == "Uppercase":
        query += " AND character GLOB '[A-Z]'"
    elif character_type == "Lowercase":
        query += " AND character GLOB '[a-z]'"

    sort_options = {
        "Newest first": "p.timestamp DESC, p.id DESC",
        "Oldest first": "p.timestamp ASC, p.id ASC",
        "Highest confidence": "p.confidence DESC, p.id DESC",
        "Lowest confidence": "p.confidence ASC, p.id ASC"
    }

    query += " ORDER BY " + sort_options[sort_order]

    with get_db_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def delete_predictions(ids):
    if not ids:
        return 0

    placeholders = ",".join("?" for _ in ids)

    with get_db_connection() as conn:
        cursor = conn.execute(
            f"DELETE FROM predictions WHERE id IN ({placeholders})",
            ids
        )
        return cursor.rowcount

def get_feedback_statistics():
    with get_db_connection() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) AS total_reviewed,
                COALESCE(SUM(is_correct), 0) AS correct,
                COUNT(*) - COALESCE(SUM(is_correct), 0) AS incorrect,
                ROUND(AVG(is_correct) * 100, 2) AS accuracy
            FROM prediction_feedback
        """).fetchone()

        return dict(row)

def get_training_samples():
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT
                p.id,
                p.image_path,
                f.actual_character
            FROM predictions p
            JOIN prediction_feedback f
                ON p.id = f.prediction_id
            WHERE p.image_path IS NOT NULL
        """).fetchall()

        return [dict(row) for row in rows]