import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom"; 

// API 호출 함수 임포트
import { predictImage, predictLink } from "../../api";

// 스타일 및 자산
import "../../css/home/HeroSection.css"; 
import bg from "../../assets/bg.png";

export default function HeroSection() {
  const canvasRef = useRef(null);
  const navigate = useNavigate(); 
  const animRef = useRef(null);

  const [dragOver, setDragOver] = useState(false);
  const [fileName, setFileName] = useState("");
  const [snsLink, setSnsLink] = useState("");
  const [isLongVideo, setIsLongVideo] = useState(false);
  const [trimRange, setTrimRange] = useState({ start: "00:00", end: "03:00" });
  
  // ✅ 분석 상태 및 진행률 관리
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);

  // ✅ 용량 제한 상수 (30MB)
  const MAX_FILE_SIZE = 30 * 1024 * 1024; 

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    let frame = 0;
    const draw = () => {
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      for (let y = 0; y < h; y += 4) {
        ctx.fillStyle = "rgba(0,0,0,0.15)";
        ctx.fillRect(0, y, w, 2);
      }
      if (frame % 20 === 0) {
        for (let i = 0; i < 3; i++) {
          ctx.fillStyle = `rgba(0,255,200,${Math.random() * 0.06})`;
          ctx.fillRect(
            Math.random() * w, Math.random() * h,
            Math.random() * 200 + 50, Math.random() * 5
          );
        }
      }
      frame++;
      animRef.current = requestAnimationFrame(draw);
    };
    draw();
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      window.removeEventListener("resize", resize);
    };
  }, []);

  const startProgressSimulation = () => {
    setProgress(0);
    return setInterval(() => {
      setProgress((prev) => {
        if (prev >= 95) return 95; 
        const increment = Math.random() * 7; 
        return Math.min(prev + increment, 95);
      });
    }, 600);
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) processFile(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  };

  const processFile = async (file) => {
    const isImage = file.type.startsWith("image/");
    const isVideo = file.type.startsWith("video/");

    if (!isImage && !isVideo) {
      alert("이미지 또는 영상 파일만 업로드 가능합니다.");
      return;
    }

    // ✅ [추가] 30MB 용량 제한 체크
    if (file.size > MAX_FILE_SIZE) {
        alert(`⚠️ 업로드 실패: 파일 용량이 너무 큽니다. (최대 30MB)\n현재 파일 용량: ${(file.size / (1024 * 1024)).toFixed(1)}MB`);
        setFileName("");
        return;
    }

    if (isVideo) {
      const video = document.createElement('video');
      video.preload = 'metadata';
      video.onloadedmetadata = async () => {
        window.URL.revokeObjectURL(video.src);
        if (video.duration > 180) {
          alert("⚠️ 업로드 실패: 3분을 초과하는 영상은 분석할 수 없습니다.");
          setFileName("");
        } else {
          setFileName(file.name);
          await uploadAndAnalyze(file);
        }
      };
      video.src = URL.createObjectURL(file);
    } else {
      setFileName(file.name);
      await uploadAndAnalyze(file);
    }
  };

  const uploadAndAnalyze = async (file) => {
    setIsAnalyzing(true); 
    const timer = startProgressSimulation();
    
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await predictImage(formData);
      if (response && response.data) {
        clearInterval(timer);
        setProgress(100); 
        setTimeout(() => {
          navigate("/result", { state: response.data });
        }, 600);
      }
    } catch (error) {
      clearInterval(timer);
      console.error("Analysis Error:", error);
      // 서버에서 413 에러(용량 초과)를 보냈을 경우에 대한 대응
      if (error.response && error.response.status === 413) {
          alert("서버에서 거부되었습니다: 파일 용량이 30MB를 초과합니다.");
      } else {
          alert("서버 연결에 실패했거나 분석 중 오류가 발생했습니다.");
      }
      setIsAnalyzing(false);
    }
  };

  const handleLinkChange = (e) => {
    const url = e.target.value;
    setSnsLink(url);
    if (url.includes("youtube.com") || url.includes("youtu.be") || url.length > 30) {
      setIsLongVideo(true);
    } else {
      setIsLongVideo(false);
    }
  };

  const handleSnsSubmit = async () => {
    if (!snsLink) {
      alert("Please paste a valid link.");
      return;
    }

    const parseTimeToSeconds = (timeStr) => {
      if (!timeStr || !timeStr.includes(':')) return 0;
      const [min, sec] = timeStr.split(':').map(Number);
      return (min || 0) * 60 + (sec || 0);
    };

    const startSec = parseTimeToSeconds(trimRange.start);
    const endSec = parseTimeToSeconds(trimRange.end);

    if (startSec >= endSec) {
      alert("❌ 시작 시간은 종료 시간보다 빨라야 합니다. 구간 설정을 확인해 주세요!");
      return;
    }

    if (endSec - startSec > 180) {
      alert("❌ 분석 가능한 최대 구간은 3분입니다. 종료 시간을 줄여주세요.");
      return;
    }

    setIsAnalyzing(true); 
    const timer = startProgressSimulation();

    try {
      const response = await predictLink({ 
        url: snsLink, 
        startTime: startSec,
        duration: endSec - startSec 
      });
      
      if (response && response.data) {
        clearInterval(timer);
        setProgress(100);
        setTimeout(() => {
          navigate("/result", { state: response.data });
        }, 600);
      }
    } catch (error) {
      clearInterval(timer);
      console.error("Link Analysis Error:", error);
      alert("링크를 분석하는 중 오류가 발생했습니다. 서버 상태를 확인해 주세요.");
      setIsAnalyzing(false);
    }
  };

  return (
    <section
      className="HeroSection"
      style={{
        backgroundImage: `url(${bg})`,
        backgroundColor: "#8ebebb",
      }}
    >
      <canvas ref={canvasRef} className="glitch-canvas" />

      <AnimatePresence>
        {isAnalyzing && (
          <motion.div 
            className="loading-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              position: 'fixed', top:0, left:0, width:'100%', height:'100%',
              background: 'rgba(0,0,0,0.9)', zIndex: 9999,
              display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center'
            }}
          >
            <div className="progress-wrapper" style={{ position: 'relative', width: '140px', height: '140px' }}>
                <svg viewBox="0 0 100 100" style={{ transform: 'rotate(-90deg)', width: '100%', height: '100%' }}>
                    <circle 
                        cx="50" cy="50" r="45" 
                        fill="transparent" stroke="rgba(255,255,255,0.1)" strokeWidth="6" 
                    />
                    <motion.circle 
                        cx="50" cy="50" r="45" 
                        fill="transparent" stroke="#00ffc8" strokeWidth="6" 
                        strokeDasharray="283"
                        animate={{ strokeDashoffset: 283 - (283 * progress) / 100 }}
                        transition={{ duration: 0.4, ease: "easeOut" }}
                        strokeLinecap="round"
                    />
                </svg>
                <div style={{
                    position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
                    color: '#00ffc8', fontSize: '1.5rem', fontWeight: 'bold', fontFamily: 'monospace'
                }}>
                    {Math.round(progress)}%
                </div>
            </div>

            <motion.p 
                animate={{ opacity: [0.4, 1, 0.4] }}
                transition={{ repeat: Infinity, duration: 1.5 }}
                style={{ color: '#00ffc8', marginTop: '30px', fontWeight: 'bold', letterSpacing: '3px', fontSize: '1.1rem' }}
            >
              AI IS ANALYZING THE MEDIA
            </motion.p>
            <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '0.85rem', marginTop: '10px' }}>
              LIME 분석 결과를 생성하고 있습니다. 잠시만 기다려주세요.
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="contents">
        <motion.div
          className="top-content"
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div className="badge">DeepEye v1.0</div>
          <h1 className="glitch-title" data-text="Real or Fake?">Real or Fake?</h1>
          <p className="subtitle">
            Upload images or videos — our AI detects deepfakes and{" "}
            <span className="accent">explains why</span>.
          </p>
        </motion.div>

        <motion.div
          className={`upload-box split ${dragOver ? "drag-active" : ""} ${isLongVideo ? "expanded" : ""}`}
          initial={{ opacity: 0, y: 60 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          <div className="split-left">
            <label className="upload-label" htmlFor="video-upload">
              <div className="icon-wrap">📁</div>
              <span className="upload-text">
                {fileName ? fileName : "Image & Video"}
              </span>
              {/* ✅ 용량 제한 가이드 텍스트 추가 */}
              <p className="upload-hint">Max 30MB</p>
            </label>
            <input
              id="video-upload"
              type="file"
              accept="image/*,video/*"
              onChange={handleFileChange}
              hidden
              disabled={isAnalyzing}
            />
          </div>

          <div className="vertical-divider"><span>OR</span></div>

          <div className="split-right">
            <div className="icon-wrap">🔗</div>
            <span className="link-title">Analyze SNS Link</span>
            
            <div className="link-input-group">
              <input 
                type="text" 
                placeholder="Paste link here..." 
                className="sns-link-input"
                value={snsLink}
                onChange={handleLinkChange}
                disabled={isAnalyzing}
                onKeyPress={(e) => e.key === 'Enter' && handleSnsSubmit()}
              />
              <button 
                className="link-submit-btn" 
                onClick={handleSnsSubmit}
                disabled={isAnalyzing}
              >
                Go
              </button>
            </div>

            <AnimatePresence>
              {isLongVideo && (
                <motion.div 
                  className="trim-settings"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                >
                  <p className="trim-notice">⚠️ Long video: Set analysis range (Max 3min)</p>
                  <div className="trim-inputs">
                    <div className="input-field">
                      <label>Start (mm:ss)</label>
                      <div className="input-with-unit">
                        <input 
                          type="text" 
                          value={trimRange.start} 
                          onChange={(e) => setTrimRange({...trimRange, start: e.target.value})}
                          placeholder="00:00"
                        />
                        <span className="unit-label">min</span>
                      </div>
                    </div>
                    <div className="input-field">
                      <label>End (mm:ss)</label>
                      <div className="input-with-unit">
                        <input 
                          type="text" 
                          value={trimRange.end} 
                          onChange={(e) => setTrimRange({...trimRange, end: e.target.value})}
                          placeholder="03:00"
                        />
                        <span className="unit-label">min</span>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>

        <div className="scroll-indicator">↓ SCROLL</div>
      </div>      
    </section>
  );
}