-- Insert cache entries (ignore duplicates)
INSERT OR IGNORE INTO operation_cache (cache_key, real_part, imag_part, unique_table_key) VALUES (?, ?, ?, ?);
