// ドラッグ&ドロップ機能の初期化
window.dragAndDropInterop = (function() {
    
    // プライベート変数
    let draggedElement = null;
    let isFromDroppedGate = false;

    // ゲートのHTMLコンテンツを取得
    function getGateContent(element) {
        if (element.classList.contains('classical-gate') && element.querySelector('svg')) {
            // SVGを含む場合はcloneNodeを使用
            return element.cloneNode(true);
        } else {
            // テキストのみの場合
            return element.textContent || element.querySelector('p')?.textContent || 'H';
        }
    }

    // ゲートタイプを取得
    function getGateType(element) {
        const classes = element.className.split(' ');
        return classes.find(cls => cls.endsWith('-gate')) || 'h-gate';
    }

    // ドラッグ開始
    function handleDragStart(e) {
        draggedElement = e.target;
        isFromDroppedGate = e.target.classList.contains('dropped-gate');
        
        e.target.classList.add('dragging');
        
        // ゲートの情報を保存
        const gateContent = getGateContent(e.target);
        const gateType = getGateType(e.target);
        
        const dragData = {
            gateType: gateType,
            isFromDropped: isFromDroppedGate,
            sourceId: isFromDroppedGate ? e.target.id || Date.now().toString() : null
        };

        // HTMLコンテンツの場合は別途処理
        if (typeof gateContent === 'string') {
            dragData.gateText = gateContent;
        } else {
            dragData.gateHTML = gateContent.outerHTML;
        }
        
        e.dataTransfer.setData('text/plain', JSON.stringify(dragData));
        e.dataTransfer.effectAllowed = isFromDroppedGate ? 'move' : 'copy';
    }

    // ドラッグ終了
    function handleDragEnd(e) {
        e.target.classList.remove('dragging');
        draggedElement = null;
        isFromDroppedGate = false;
    }

    // ドロップゾーンの初期化
    function initializeDropZones() {
        console.log('Initializing drop zones...');
        
        // 量子ビット線のドロップゾーン
        const qubitLines = document.querySelectorAll('.qubit-line');
        console.log(`Found ${qubitLines.length} qubit lines`);
        
        qubitLines.forEach((line, index) => {
            console.log(`Initializing qubit line ${index}:`, line);
            
            // 既存のイベントリスナーを削除
            line.removeEventListener('dragover', handleQubitDragOver);
            line.removeEventListener('drop', handleQubitDrop);
            line.removeEventListener('dragenter', handleQubitDragEnter);
            line.removeEventListener('dragleave', handleQubitDragLeave);
            
            // 新しいイベントリスナーを追加
            line.addEventListener('dragover', handleQubitDragOver);
            line.addEventListener('drop', handleQubitDrop);
            line.addEventListener('dragenter', handleQubitDragEnter);
            line.addEventListener('dragleave', handleQubitDragLeave);
            
            // ドロップゾーンが正しく設定されているかテスト
            line.style.minHeight = '20px';
            line.style.position = 'relative';
        });

        // ゴミ箱のドロップゾーン
        const trashZone = document.querySelector('.trash-zone');
        if (trashZone) {
            trashZone.removeEventListener('dragover', handleTrashDragOver);
            trashZone.removeEventListener('drop', handleTrashDrop);
            trashZone.removeEventListener('dragenter', handleTrashDragEnter);
            trashZone.removeEventListener('dragleave', handleTrashDragLeave);
            
            trashZone.addEventListener('dragover', handleTrashDragOver);
            trashZone.addEventListener('drop', handleTrashDrop);
            trashZone.addEventListener('dragenter', handleTrashDragEnter);
            trashZone.addEventListener('dragleave', handleTrashDragLeave);
        }
    }

    // 量子ビット線のドラッグイベント
    function handleQubitDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        console.log('Drag over qubit line');
        e.dataTransfer.dropEffect = isFromDroppedGate ? 'move' : 'copy';
    }

    function handleQubitDragEnter(e) {
        e.preventDefault();
        e.stopPropagation();
        console.log('Drag enter qubit line');
        const target = e.target.closest('.qubit-line') || e.target;
        target.style.backgroundColor = '#512BD4';
        target.style.height = '3px';
    }

    function handleQubitDragLeave(e) {
        e.stopPropagation();
        console.log('Drag leave qubit line');
        const target = e.target.closest('.qubit-line') || e.target;
        // 子要素からのleaveイベントを無視
        if (!target.contains(e.relatedTarget)) {
            target.style.backgroundColor = '';
            target.style.height = '1px';
        }
    }

    function handleQubitDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        console.log('Drop on qubit line detected!');
        
        const target = e.target.closest('.qubit-line') || e.target;
        target.style.backgroundColor = '';
        target.style.height = '1px';
        
        try {
            const dragDataString = e.dataTransfer.getData('text/plain');
            console.log('Drag data received:', dragDataString);
            
            if (!dragDataString) {
                console.error('No drag data found');
                return;
            }
            
            const data = JSON.parse(dragDataString);
            console.log('Parsed drag data:', data);
            
            const qubitLine = target;
            
            // 移動の場合は元の要素を削除
            if (data.isFromDropped && draggedElement) {
                draggedElement.remove();
            }
            
            // 新しいゲート要素を作成
            const newGate = document.createElement('div');
            newGate.className = data.gateType + ' dropped-gate';
            newGate.draggable = true;
            newGate.id = 'gate-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
            
            // コンテンツを設定
            if (data.gateHTML) {
                newGate.innerHTML = data.gateHTML;
            } else {
                newGate.textContent = data.gateText;
            }
            
            // スタイルを設定
            newGate.style.position = 'absolute';
            newGate.style.transform = 'translateY(-50%)';
            newGate.style.cursor = 'grab';
            newGate.style.zIndex = '10';
            
            // ドロップ位置を計算（グリッドにスナップ）
            const rect = qubitLine.getBoundingClientRect();
            const dropX = e.clientX - rect.left;
            const gridSize = 50; // CSSのbackground-sizeと同じ
            const snappedX = Math.round(dropX / gridSize) * gridSize;
            newGate.style.left = snappedX + 'px';
            
            // イベントリスナーを追加
            newGate.addEventListener('dragstart', handleDragStart);
            newGate.addEventListener('dragend', handleDragEnd);
            
            // ダブルクリックで削除
            newGate.addEventListener('dblclick', function() {
                newGate.remove();
            });
            
            // 量子ビット線に配置
            qubitLine.style.position = 'relative';
            qubitLine.appendChild(newGate);
            
        } catch (error) {
            console.error('ドロップ処理でエラーが発生しました:', error);
        }
    }

    // ゴミ箱のドラッグイベント
    function handleTrashDragOver(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
    }

    function handleTrashDragEnter(e) {
        e.preventDefault();
        e.target.closest('.trash-zone').classList.add('drag-over');
    }

    function handleTrashDragLeave(e) {
        const trashZone = e.target.closest('.trash-zone');
        if (trashZone && !trashZone.contains(e.relatedTarget)) {
            trashZone.classList.remove('drag-over');
        }
    }

    function handleTrashDrop(e) {
        e.preventDefault();
        e.target.closest('.trash-zone').classList.remove('drag-over');
        
        try {
            const data = JSON.parse(e.dataTransfer.getData('text/plain'));
            
            // ドロップされたゲートのみ削除可能
            if (data.isFromDropped && draggedElement) {
                draggedElement.remove();
            }
            
        } catch (error) {
            console.error('ゴミ箱でのドロップ処理でエラーが発生しました:', error);
        }
    }

    // パレットのゲートを初期化
    function initializePaletteGates() {
        console.log('Initializing palette gates...');
        
        const gates = document.querySelectorAll('.h-gate, .classical-gate, .phase-gate, .other-gate');
        console.log(`Found ${gates.length} gates in palette`);
        
        gates.forEach((gate, index) => {
            console.log(`Initializing gate ${index}:`, gate);
            
            // 既存のイベントリスナーを削除
            gate.removeEventListener('dragstart', handleDragStart);
            gate.removeEventListener('dragend', handleDragEnd);
            
            // 新しいイベントリスナーを追加
            gate.addEventListener('dragstart', handleDragStart);
            gate.addEventListener('dragend', handleDragEnd);
            
            // draggableが正しく設定されているか確認
            if (!gate.draggable) {
                gate.draggable = true;
            }
        });
    }

    // パブリックAPI
    return {
        initializeDragAndDrop: function() {
            console.log('Starting drag and drop initialization...');
            initializePaletteGates();
            initializeDropZones();
            console.log('Drag and drop initialization completed');
        },

        reinitialize: function() {
            console.log('Reinitializing drag and drop...');
            this.initializeDragAndDrop();
        },

        // デバッグ用関数
        debug: function() {
            console.log('=== Drag and Drop Debug Info ===');
            const qubitLines = document.querySelectorAll('.qubit-line');
            const gates = document.querySelectorAll('.h-gate, .classical-gate, .phase-gate, .other-gate');
            console.log(`Qubit lines found: ${qubitLines.length}`);
            console.log(`Gates found: ${gates.length}`);
            console.log(`Dragged element:`, draggedElement);
            console.log(`Is from dropped gate:`, isFromDroppedGate);
            
            qubitLines.forEach((line, index) => {
                console.log(`Qubit line ${index} events:`, {
                    dragover: line.ondragover !== null,
                    drop: line.ondrop !== null,
                    dragenter: line.ondragenter !== null,
                    dragleave: line.ondragleave !== null
                });
            });
        }
    };
})();