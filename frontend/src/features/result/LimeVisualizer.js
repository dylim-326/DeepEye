import React from 'react';
import "../../css/result/LimeVisualizer.css";

export default function LimeVisualizer({ data }) {
    const limeImageUrl = data?.lime_url;
    const result = data?.result; // 'FAKE' or 'REAL'
    const isDiscrepancy = data?.is_discrepancy; // Model consensus status

    // Situational customized report generation logic
    const getReportContent = () => {
        // 1. Model Discrepancy (Prioritizing primary EfficientNet result)
        if (isDiscrepancy) {
            return (
                <>
                    Data distortion identified by the primary engine (EfficientNet) served as the decisive basis for the prediction. 
                    The <span style={{ color: '#FFD700', fontWeight: 'bold' }}>yellow</span> region indicates the primary focus area, 
                    displaying only the primary model's attribution due to cross-model disagreement.
                </>
            );
        }

        // 2. Consensus: FAKE
        if (result === "FAKE") {
            return (
                <>
                    Unnatural artifact patterns within the visualized image were captured as decisive evidence of manipulation. 
                    Yellow (EffNet) and Orange (MesoNet) represent individual suspicion areas, while the 
                    <span style={{ color: '#FF4D4D', fontWeight: 'bold' }}> red</span> region highlights high-risk segments where both models jointly identified manipulation traces.
                </>
            );
        }

        // 3. Consensus: REAL
        return (
            <>
                Natural facial characteristics without artificial manipulation traces were verified across the image. 
                Yellow (EffNet) and Orange (MesoNet) indicate areas of confirmed data integrity, while the 
                <span style={{ color: '#FF4D4D', fontWeight: 'bold' }}> red</span> region indicates confidence zones where both models mutually verified authentic capture features.
            </>
        );
    };

    return (
        <section className="LimeVisualizer">
            <div className="section-title">
                <span className="badge">LIME XAI Analysis</span>
                <h2>Visual Interpretation Map</h2>
            </div>
            
            <div className="visual-layout">
                <div className="media-box">
                    {limeImageUrl ? (
                        <img 
                            src={limeImageUrl} 
                            alt="LIME Heatmap Analysis" 
                            className="analysis-image"
                        />
                    ) : (
                        <div className="placeholder-container">
                            <div className="spinner"></div>
                            <p>Generating LIME Heatmap...</p>
                        </div>
                    )}
                </div>

                <div className="explanation-box">
                    <p>
                        <span className="accent-text">Analysis Report:</span><br />
                        {getReportContent()}
                    </p>
                </div>
            </div>
        </section>
    );
}