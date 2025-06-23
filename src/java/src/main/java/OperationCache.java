import java.util.concurrent.ForkJoinPool;

import org.graalvm.nativeimage.IsolateThread;
import org.graalvm.nativeimage.UnmanagedMemory;
import org.graalvm.nativeimage.c.function.CEntryPoint;
import org.graalvm.word.Pointer;
import org.graalvm.word.WordFactory;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;

public class OperationCache {
    private static final Cache<Long, OperationResult> CACHE;

    static {

        CACHE = Caffeine.newBuilder()
                .maximumSize(1_048_576)
                .initialCapacity(262_144)
                .recordStats()
                .build();
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

    public static void main(String[] args) {
        System.out.println("OperationCache initialized");
    }
}