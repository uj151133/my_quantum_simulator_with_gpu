# キャッシュの初期化
cache = {}

# Fiberを使ったキャッシュ設定の関数
def set_cache(fiber, key, value)
    fiber.resume(:set, key, value)
    fiber.resume  # :okを受け取る
end

# Fiberを使ったキャッシュ取得の関数
def get_cache(fiber, key)
    fiber.resume(:get, key)
    fiber.resume  # 実際の値を受け取る
end

# Fiberの初期化
cache_fiber = Fiber.new do
    loop do
        command, key, value = Fiber.yield

        case command
        when :get
            Fiber.yield cache[key]
        when :set
            cache[key] = value
            Fiber.yield :ok
        end
    end
end

# キャッシュに値を設定する
set_cache(cache_fiber, :foo, "bar")
set_cache(cache_fiber, :baz, "qux")

# キャッシュから値を取得する
result = get_cache(cache_fiber, :foo)
puts "Cache get for :foo -> #{result}"

result = get_cache(cache_fiber, :baz)
puts "Cache get for :baz -> #{result}"
