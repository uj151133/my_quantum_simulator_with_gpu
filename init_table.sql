-- Table initialization
CREATE TABLE IF NOT EXISTS operation_cache (
    cache_key INTEGER PRIMARY KEY,
    real_part REAL NOT NULL,
    imag_part REAL NOT NULL,
    unique_table_key INTEGER NOT NULL,
    access_count INTEGER DEFAULT 1
);
