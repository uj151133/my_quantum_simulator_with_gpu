import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.Statement;

import org.graalvm.nativeimage.IsolateThread;
import org.graalvm.nativeimage.UnmanagedMemory;
import org.graalvm.nativeimage.c.function.CEntryPoint;
import org.graalvm.word.Pointer;
import org.graalvm.word.WordFactory;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;

public class OperationCache {
    private static final Cache<Long, OperationResult> CACHE;
    private static final String DB_PATH = "./bible.db";
    private static final String INIT_TABLE_SQL_PATH = "./init_table.sql";
    private static final String INSERT_CACHE_SQL_PATH = "./insert_cache.sql";

    static {
        CACHE = Caffeine.newBuilder()
                .maximumSize(1_048_576)
                .initialCapacity(262_144)
                .recordStats()
                .build();
    }
    
    private static void initializeDatabase() {
        try {
            // データベースファイルが既に存在する場合はスキップ
            if (Files.exists(Paths.get(DB_PATH))) {
                System.out.println("OperationCache: SQLite database already exists at " + DB_PATH);
                return;
            }
            
            Class.forName("org.sqlite.JDBC");
            
            try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + DB_PATH)) {
                String initTableSql = readSqlFile(INIT_TABLE_SQL_PATH);
                try (Statement stmt = conn.createStatement()) {
                    stmt.execute(initTableSql);
                    System.out.println("OperationCache: SQLite database initialized at " + DB_PATH);
                }
            }
        } catch (Exception e) {
            System.err.println("OperationCache: Failed to initialize SQLite database: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static String readSqlFile(String filePath) throws IOException {
        return new String(Files.readAllBytes(Paths.get(filePath)));
    }

    private OperationCache() {
        throw new AssertionError("Utility class");
    }

    public static void insert(long key, OperationResult result) {
        CACHE.put(key, result);
    }

    @CEntryPoint(name = "cacheInsert")
    public static void nativeInsert(IsolateThread thread, long key, double real, double imag, long uniqueTableKey) {
        CACHE.put(key, new OperationResult(real, imag, uniqueTableKey));
    }

    public static OperationResult find(long key) {
        return CACHE.getIfPresent(key);
    }

    @CEntryPoint(name = "cacheFind")
    public static Pointer nativeFind(IsolateThread thread, long key) {
        OperationResult result = CACHE.getIfPresent(key);
        if (result == null) {
            return WordFactory.nullPointer();
        }
        
        Pointer ptr = UnmanagedMemory.malloc(24);

        ptr.writeDouble(0, result.real());
        ptr.writeDouble(8, result.imag());
        ptr.writeLong(16, result.uniqueTableKey());
        return ptr;
    }

    public boolean contains(long key) {
        return CACHE.getIfPresent(key) != null;
    }
    
    public void invalidate(long key) {
        CACHE.invalidate(key);
    }
    
    public void clear() {
        CACHE.invalidateAll();
    }
    
    public long size() {
        return CACHE.estimatedSize();
    }

    public void printStats() {
        System.out.println("Cache size: " + CACHE.estimatedSize());
    }

    public static void printAllEntries() {
        System.out.println("=== OperationCache All Entries ===");
        System.out.println("Total entries: " + CACHE.estimatedSize());
        System.out.println("----------------------------------------");
        
        long count = 0;
        for (var entry : CACHE.asMap().entrySet()) {
            Long key = entry.getKey();
            OperationResult result = entry.getValue();
            
            System.out.printf("Entry %d: Key=%d, Real=%.6f, Imag=%.6f, UniqueTableKey=%d%n", 
                ++count, key, result.real(), result.imag(), result.uniqueTableKey());
        }
        
        if (count == 0) {
            System.out.println("Cache is empty.");
        }
        
        System.out.println("----------------------------------------");
        System.out.println("Cache statistics:");
        System.out.println(CACHE.stats());
    }

    public static void main(String[] args) {
        System.out.println("OperationCache initialized");
    }
    
    @CEntryPoint(name = "saveCacheToSQLite")
    public static void saveCacheToSQLite(IsolateThread thread) {
        System.out.println("OperationCache: Starting cache save to SQLite...");
        
        initializeDatabase();
        
        try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + DB_PATH)) {
            String insertSql = readSqlFile(INSERT_CACHE_SQL_PATH);
            
            try (PreparedStatement pstmt = conn.prepareStatement(insertSql)) {
                long savedCount = 0;
                
                for (var entry : CACHE.asMap().entrySet()) {
                    Long key = entry.getKey();
                    OperationResult result = entry.getValue();
                    
                    pstmt.setLong(1, key);
                    pstmt.setDouble(2, result.real());
                    pstmt.setDouble(3, result.imag());
                    pstmt.setLong(4, result.uniqueTableKey());
                    
                    pstmt.executeUpdate();
                    savedCount++;
                }
                
                System.out.println("OperationCache: Successfully saved " + savedCount + " entries to SQLite");
                System.out.println("OperationCache: Database file: " + DB_PATH);
                System.out.println("OperationCache: Total cache size: " + CACHE.estimatedSize());
                
            }
        } catch (Exception e) {
            System.err.println("OperationCache: Failed to save cache to SQLite: " + e.getMessage());
            e.printStackTrace();
        }
    }
}