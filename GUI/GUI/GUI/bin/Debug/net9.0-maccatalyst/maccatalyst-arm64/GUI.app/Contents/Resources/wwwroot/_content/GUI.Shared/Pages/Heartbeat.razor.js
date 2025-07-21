// Heartbeatページのダッシュボード機能
window.heartbeatInterop = {
    // リアルタイムでデータを更新する機能
    startRealTimeUpdate: function(dotNetObjectReference) {
        console.log('Starting real-time heartbeat updates...');
        
        // 3秒ごとにデータを更新（ダッシュボードなのでより頻繁に）
        this.updateInterval = setInterval(() => {
            try {
                dotNetObjectReference.invokeMethodAsync('UpdateHeartbeatData');
            } catch (error) {
                console.error('Failed to update heartbeat data:', error);
            }
        }, 3000);
    },

    // リアルタイム更新を停止
    stopRealTimeUpdate: function() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
            console.log('Stopped real-time heartbeat updates');
        }
    },

    // メトリクスカードのアニメーション
    animateMetrics: function() {
        const metricCards = document.querySelectorAll('.metric-card');
        metricCards.forEach((card, index) => {
            // 順次アニメーション
            setTimeout(() => {
                card.style.transform = 'translateY(-4px)';
                card.style.boxShadow = '0 6px 20px rgba(0, 0, 0, 0.15)';
                
                setTimeout(() => {
                    card.style.transform = 'translateY(0)';
                    card.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.1)';
                }, 300);
            }, index * 100);
        });
    },

    // ステータスカードのアニメーション
    animateStatusChange: function(isHealthy) {
        const statusCard = document.querySelector('.status-card');
        if (statusCard) {
            if (isHealthy) {
                statusCard.classList.add('status-pulse-healthy');
            } else {
                statusCard.classList.add('status-pulse-unhealthy');
            }
            
            // アニメーション後にクラスを削除
            setTimeout(() => {
                statusCard.classList.remove('status-pulse-healthy', 'status-pulse-unhealthy');
            }, 1000);
        }
    },

    // メトリクスの数値アニメーション
    animateNumbers: function() {
        const metricValues = document.querySelectorAll('.metric-value');
        metricValues.forEach(element => {
            element.style.transform = 'scale(1.05)';
            element.style.transition = 'transform 0.3s ease';
            
            setTimeout(() => {
                element.style.transform = 'scale(1)';
            }, 300);
        });
    },

    // プログレスバーのアニメーション
    animateProgressBars: function() {
        const progressBars = document.querySelectorAll('.metric-fill');
        progressBars.forEach(bar => {
            const width = bar.style.width;
            bar.style.width = '0%';
            bar.style.transition = 'width 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
            
            setTimeout(() => {
                bar.style.width = width;
            }, 100);
        });
    },

    // メッセージの更新アニメーション
    animateMessage: function() {
        const messageText = document.querySelector('.message-text');
        if (messageText) {
            messageText.style.opacity = '0.5';
            messageText.style.transform = 'translateX(-10px)';
            messageText.style.transition = 'all 0.3s ease';
            
            setTimeout(() => {
                messageText.style.opacity = '1';
                messageText.style.transform = 'translateX(0)';
            }, 150);
        }
    },

    // 初期化メソッド
    initialize: function() {
        console.log('Initializing heartbeat dashboard...');
        
        // ページが表示されたときにハートビートアニメーションを開始
        const heartIcon = document.querySelector('.heart-icon');
        if (heartIcon) {
            heartIcon.style.animation = 'heartbeat 1.5s ease-in-out infinite';
        }

        // 初期アニメーション
        this.animateMetrics();
        this.animateProgressBars();
    },

    // 全体的な更新アニメーション
    updateAnimations: function() {
        this.animateNumbers();
        this.animateProgressBars();
        this.animateMessage();
    },

    // クリーンアップメソッド
    cleanup: function() {
        this.stopRealTimeUpdate();
        console.log('Heartbeat dashboard cleanup completed');
    }
};

// ページロード時に初期化
document.addEventListener('DOMContentLoaded', function() {
    if (window.heartbeatInterop) {
        window.heartbeatInterop.initialize();
    }
});
