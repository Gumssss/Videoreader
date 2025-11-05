document.addEventListener('DOMContentLoaded', function() {
    const cameraView = document.getElementById('camera-view');
    const cameraElement = document.getElementById('camera');
    const canvasElement = document.getElementById('canvas');
    const startCameraBtn = document.getElementById('start-camera');
    const captureBtn = document.getElementById('capture-btn');
    const cameraFileInput = document.getElementById('camera-file');
    const cameraForm = document.getElementById('camera-form');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    let stream = null;
    
    // 启动摄像头
    startCameraBtn.addEventListener('click', async function() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    facingMode: 'environment',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false 
            });
            
            cameraElement.srcObject = stream;
            cameraElement.style.display = 'block';
            cameraView.style.display = 'none';
            startCameraBtn.disabled = true;
            captureBtn.disabled = false;
            
            // 调整摄像头视图大小
            cameraElement.onloadedmetadata = () => {
                canvasElement.width = cameraElement.videoWidth;
                canvasElement.height = cameraElement.videoHeight;
            };
        } catch (err) {
            console.error("Camera error:", err);
            alert("无法访问摄像头: " + err.message);
        }
    });
    
    // 拍照
    captureBtn.addEventListener('click', function() {
        // 绘制当前帧到canvas
        canvasElement.getContext('2d').drawImage(cameraElement, 0, 0, canvasElement.width, canvasElement.height);
        
        // 停止摄像头
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        
        // 显示拍照结果
        cameraElement.style.display = 'none';
        canvasElement.style.display = 'block';
        captureBtn.style.display = 'none';
        startCameraBtn.style.display = 'none';
        
        // 将canvas转换为Blob并设置为文件输入
        canvasElement.toBlob(function(blob) {
            const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            cameraFileInput.files = dataTransfer.files;
            
            // 显示分析按钮
            cameraForm.style.display = 'block';
        }, 'image/jpeg', 0.95);
    });
    
    // 分析照片
    analyzeBtn.addEventListener('click', function() {
        // 可以在这里添加加载指示器
        this.innerHTML = this.dataset.loadingText || 'Analyzing...';
    });
    
    // 页面卸载时停止摄像头
    window.addEventListener('beforeunload', function() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    });
});