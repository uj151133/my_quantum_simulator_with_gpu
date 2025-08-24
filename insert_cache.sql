-- Insert cache entries with timestamp (ignore duplicates, update timestamp on conflict)
INSERT INTO operation_cache (cache_key, real_part, imag_part, unique_table_key, created_at, updated_at) 
VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT(cache_key) DO UPDATE SET 
    access_count = access_count + 1,
    updated_at = CURRENT_TIMESTAMP;
