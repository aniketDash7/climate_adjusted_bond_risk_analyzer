import { useState, useEffect } from 'react'
import { MapContainer, TileLayer, Marker, Popup, Polyline } from 'react-leaflet'
import L from 'leaflet'
import axios from 'axios'
import 'leaflet/dist/leaflet.css'
import './index.css'

const createBubbleIcon = (riskScore, overrideColor = null) => {
  const size = Math.max(18, riskScore * 42);
  const scoreText = Math.round(riskScore * 10);
  
  // Custom styles for pulsing alert marker
  const style = overrideColor 
    ? `background-color: ${overrideColor}; color: white; border: 2px solid white; box-shadow: 0 0 15px ${overrideColor}; animation: pulse 1s infinite;` 
    : '';

  return L.divIcon({
    html: `<span style="${style}">${scoreText}</span>`,
    className: 'marker-bubble',
    iconSize: [size, size],
    iconAnchor: [size/2, size/2]
  });
};

// Mini bar component for hazard breakdown
const HazardBar = ({ label, value, color, emoji }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
    <span style={{ width: '14px', fontSize: '12px' }}>{emoji}</span>
    <span style={{ width: '60px', fontSize: '0.75rem', color: '#666' }}>{label}</span>
    <div style={{ flex: 1, height: '6px', background: '#eee', borderRadius: '3px', overflow: 'hidden' }}>
      <div style={{ width: `${value * 100}%`, height: '100%', background: color, borderRadius: '3px' }}></div>
    </div>
    <span style={{ width: '35px', fontSize: '0.75rem', fontWeight: 700, textAlign: 'right' }}>{(value * 100).toFixed(0)}%</span>
  </div>
);

function App() {
  const [bonds, setBonds] = useState([])
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [alerts, setAlerts] = useState([])

  // WebSocket Connection
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:5001/ws/alerts')
    
    ws.onopen = () => console.log('Connected to Alert Engine')
    ws.onmessage = (event) => {
      const alertData = JSON.parse(event.data)
      console.log('Received Alert:', alertData)
      setAlerts(prev => [alertData, ...prev])
    }
    
    return () => ws.close()
  }, [])

  useEffect(() => {
    const fetchData = async () => {
      try {
        const bondRes = await axios.get('http://localhost:5000/api/bonds')
        const statRes = await axios.get('http://localhost:5000/api/stats')
        setBonds(bondRes.data)
        setStats(statRes.data)
        setLoading(false)
      } catch (error) {
        console.error("Error fetching data", error)
      }
    }
    fetchData()
  }, [])

  const mapCenter = [37.2, -119.5]

  if (loading) return (
    <div style={{ display: 'flex', height: '100vh', alignItems: 'center', justifyContent: 'center', fontWeight: 900, textTransform: 'uppercase' }}>
      Loading Climate Risk Engine...
    </div>
  )

  return (
    <div className="app-container">
      <div className="sidebar">
        <h1 className="title">Climate Risk</h1>
        <p className="subtitle">Multi-Hazard Municipal Bond Analysis</p>

        {stats && (
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-label">Portfolio Exposure</div>
              <div className="stat-value">${(stats.total_exposure / 1e9).toFixed(2)}B</div>
            </div>

            <div className="stat-card">
              <div className="stat-label">Climate Value-at-Risk</div>
              <div className="stat-value danger">${(stats.total_var / 1e6).toFixed(1)}M</div>
            </div>

            <div className="stat-card">
              <div className="stat-label">Multi-Hazard Composite</div>
              <div className="stat-value">{(stats.avg_risk_score * 100).toFixed(0)}%</div>
            </div>

            <div className="stat-card">
              <div className="stat-label tooltip" title="Scores are normalized indices from 0 (Safe) to 1 (Extreme Risk)">Hazard Breakdown (Avg)</div>
              <div style={{ marginTop: '8px' }}>
                <HazardBar label="RF Fire" value={stats.avg_wildfire || 0} color="#e63946" />
                <HazardBar label="DL Path" value={stats.avg_dl_prob || 0} color="#d62828" />
                <HazardBar label="Flood" value={stats.avg_flood || 0} color="#457b9d" />
                <HazardBar label="Quake" value={stats.avg_earthquake || 0} color="#6d6875" />
                <HazardBar label="NDVI" value={stats.avg_ndvi || 0} color="#2d6a4f" />
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-label tooltip" title="Basis Points (bps). 100 bps = 1.00% yield penalty modeled for physical risk.">Avg Climate Spread</div>
              <div className="stat-value">+{(stats.avg_spread_bps || 0).toFixed(0)} bps</div>
            </div>
          </div>
        )}

        {/* Real-time Alerts Panel */}
        {alerts.length > 0 && (
          <div className="alerts-panel">
            <h2 style={{ fontSize: '0.8rem', color: '#e63946', textTransform: 'uppercase', marginBottom: '8px' }}>
              Live Active Fire Alerts
            </h2>
            <div className="alerts-list">
              {alerts.slice(0, 5).map((alert, i) => (
                <div key={i} className="alert-item">
                  <strong>{alert.issuer}</strong><br/>
                  Active fire detected {alert.distance_km}km away!
                </div>
              ))}
            </div>
          </div>
        )}

        <div style={{ marginTop: 'auto', fontSize: '0.6rem', color: '#999', textTransform: 'uppercase', letterSpacing: '0.1em', lineHeight: '1.6' }}>
          Sources: FIRMS &middot; FEMA &middot; Sentinel-2<br/>
          Models: RF + PyTorch CNN/LSTM (0.93 AUC)
        </div>
      </div>

      <div className="map-container">
        <MapContainer
          center={mapCenter}
          zoom={6.5}
          scrollWheelZoom={true}
          style={{ height: "100%", width: "100%" }}
          zoomControl={false}
        >
          <TileLayer
            attribution='&copy; <a href="https://carto.com/attributions">CARTO</a>'
            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          />

          {bonds.map((bond) => {
            // Check if this bond has an active alert
            const hasAlert = alerts.some(a => a.bond_id === bond.bond_id);
            const markerColor = hasAlert ? '#ff0000' : null;
            
            return (
              <Marker
                key={bond.bond_id}
                position={[bond.lat, bond.lon]}
                icon={createBubbleIcon(bond.composite_score || bond.risk_score, markerColor)}
              >
                <Popup className="custom-popup">
                <div className="popup-content">
                  <div className="popup-title">{(bond.issuer || '').toUpperCase()}</div>

                  <div style={{ margin: '8px 0', padding: '8px', background: '#f9f9f9', borderRadius: '4px' }}>
                    <HazardBar label="RF Fire" value={bond.wildfire_score || 0}         color="#e63946" />
                    <HazardBar label="DL Path" value={bond.dl_fire_prob || 0}           color="#d62828" />
                    <HazardBar label="Flood" value={bond.flood_score || 0}              color="#457b9d" />
                    <HazardBar label="Quake" value={bond.earthquake_score || 0}         color="#6d6875" />
                    <HazardBar label="NDVI" value={bond.ndvi || 0}                      color="#2d6a4f" />
                  </div>

                  <div className="popup-row">
                    <span className="popup-label">Composite Risk</span>
                    <span className="popup-value" style={{ color: '#e63946' }}>
                      {((bond.composite_score || bond.risk_score) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="popup-row">
                    <span className="popup-label tooltip" title="Estimated basis point yield premium required to offset climate risk.">Climate Spread</span>
                    <span className="popup-value">+{(bond.climate_spread_bps || 0).toFixed(0)} bps</span>
                  </div>
                  <div className="popup-row" style={{ borderTop: '1px solid #ddd', marginTop: '4px', paddingTop: '4px' }}>
                    <span className="popup-label">Fair Yield</span>
                    <span className="popup-value">{(bond.fair_value_yield || 0).toFixed(2)}%</span>
                  </div>
                </div>
              </Popup>
            </Marker>
            );
          })}

          {/* Render Fire Paths for active alerts */}
          {alerts.map((alert, idx) => {
            const targetBond = bonds.find(b => b.bond_id === alert.bond_id);
            if (!targetBond) return null;
            return (
              <Polyline
                key={`path-${idx}`}
                positions={[
                  [alert.fire_lat, alert.fire_lon],
                  [targetBond.lat, targetBond.lon]
                ]}
                className="fire-path-animation"
                pathOptions={{ dashArray: '8 12', color: '#ff0000', weight: 4 }}
              />
            );
          })}

          <div className="legend-pill">
            <div className="legend-item">
              <span>Risk Composites = Fire (NDVI-adj) | Flood | Quake</span>
            </div>
            <div style={{ color: '#ccc' }}>|</div>
            <div className="legend-item">
              <span>FIRMS &middot; FEMA NRI &middot; Sentinel-2</span>
            </div>
          </div>
        </MapContainer>
      </div>
    </div>
  )
}

export default App
