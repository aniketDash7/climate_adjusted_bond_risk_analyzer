import { useState, useEffect } from 'react'
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet'
import L from 'leaflet'
import axios from 'axios'
import 'leaflet/dist/leaflet.css'
import './index.css'

// Custom DivIcon creator for the red "bubble" markers
const createBubbleIcon = (riskScore) => {
  const size = Math.max(16, riskScore * 40); // Size scales with risk
  const scoreText = Math.round(riskScore * 10); // Simple 1-10 ranking inside bubble

  return L.divIcon({
    html: `<span>${scoreText}</span>`,
    className: 'marker-bubble',
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2]
  });
};

function App() {
  const [bonds, setBonds] = useState([])
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Backend now running on 5000 as per latest updates
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
        <p className="subtitle">ML-Powered Municipal Bond Portfolio Analysis</p>

        {stats && (
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-label">Total Portfolio Asset Value</div>
              <div className="stat-value">${(stats.total_exposure / 1e9).toFixed(2)}B</div>
            </div>

            <div className="stat-card">
              <div className="stat-label">Wildfire Value-at-Risk</div>
              <div className="stat-value danger">
                ${(stats.total_var / 1e6).toFixed(1)}M
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-label">Avg Risk Intensity</div>
              <div className="stat-value">{(stats.avg_risk_score * 100).toFixed(0)}%</div>
            </div>

            <div className="stat-card">
              <div className="stat-label">Weighted Yield Spread</div>
              <div className="stat-value">+{(stats.avg_spread_bps || 0).toFixed(0)} bps</div>
            </div>
          </div>
        )}

        <div style={{ marginTop: 'auto', fontSize: '0.65rem', color: '#999', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
          Data Source: NASA FIRMS (VIIRS 375m) & OpenMeteo<br />
          Model: RandomForest (0.93 AUC)
        </div>
      </div>

      <div className="map-container">
        <MapContainer
          center={mapCenter}
          zoom={6.5}
          scrollWheelZoom={true}
          style={{ height: "100%", width: "100%" }}
          zoomControl={false} // Cleaner look
        >
          {/* Grayscale Carto Positron Tiles */}
          <TileLayer
            attribution='&copy; <a href="https://carto.com/attributions">CARTO</a>'
            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          />

          {bonds.map((bond) => (
            <Marker
              key={bond.bond_id}
              position={[bond.lat, bond.lon]}
              icon={createBubbleIcon(bond.risk_score)}
            >
              <Popup className="custom-popup">
                <div className="popup-content">
                  <div className="popup-title">{bond.issuer.toUpperCase()}</div>
                  <div className="popup-row">
                    <span className="popup-label">Rating</span>
                    <span className="popup-value">{bond.rating}</span>
                  </div>
                  <div className="popup-row">
                    <span className="popup-label">ML Risk Score</span>
                    <span className="popup-value" style={{ color: '#e63946' }}>{(bond.risk_score * 100).toFixed(1)}%</span>
                  </div>
                  <div className="popup-row">
                    <span className="popup-label">Climate Spread</span>
                    <span className="popup-value">+{bond.climate_spread_bps.toFixed(0)} bps</span>
                  </div>
                  <div className="popup-row" style={{ borderTop: '1px solid #ddd', marginTop: '4px', paddingTop: '4px' }}>
                    <span className="popup-label">Fair Yield</span>
                    <span className="popup-value">{bond.fair_value_yield.toFixed(2)}%</span>
                  </div>
                </div>
              </Popup>
            </Marker>
          ))}

          {/* Floating UI Elements */}
          <div className="legend-pill">
            <div className="legend-item">
              <div className="legend-color"></div>
              <span>Wildfire Risk Concentration (ML Predict)</span>
            </div>
            <div style={{ color: '#ccc' }}>|</div>
            <div className="legend-item">
              <span>Bubble size = Exposure × Risk</span>
            </div>
          </div>
        </MapContainer>
      </div>
    </div>
  )
}

export default App
