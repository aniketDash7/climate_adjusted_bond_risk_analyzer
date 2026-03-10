import { useState, useEffect } from 'react'
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet'
import L from 'leaflet'
import axios from 'axios'
import 'leaflet/dist/leaflet.css'
import './index.css'

const createBubbleIcon = (riskScore) => {
  const size = Math.max(18, riskScore * 42);
  const scoreText = Math.round(riskScore * 10);
  return L.divIcon({
    html: `<span>${scoreText}</span>`,
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
              <div className="stat-label">Hazard Breakdown (Avg)</div>
              <div style={{ marginTop: '8px' }}>
                <HazardBar label="Fire" value={stats.avg_wildfire || 0} color="#e63946" emoji="🔥" />
                <HazardBar label="Flood" value={stats.avg_flood || 0} color="#457b9d" emoji="🌊" />
                <HazardBar label="Quake" value={stats.avg_earthquake || 0} color="#6d6875" emoji="🌍" />
                <HazardBar label="NDVI" value={stats.avg_ndvi || 0} color="#2d6a4f" emoji="🌿" />
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-label">Avg Climate Spread</div>
              <div className="stat-value">+{(stats.avg_spread_bps || 0).toFixed(0)} bps</div>
            </div>
          </div>
        )}

        <div style={{ marginTop: 'auto', fontSize: '0.6rem', color: '#999', textTransform: 'uppercase', letterSpacing: '0.1em', lineHeight: '1.6' }}>
          Sources: NASA FIRMS &middot; FEMA NRI &middot; Sentinel-2<br/>
          Model: RF (0.93 AUC) &middot; NDVI-adjusted Composite
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

          {bonds.map((bond) => (
            <Marker
              key={bond.bond_id}
              position={[bond.lat, bond.lon]}
              icon={createBubbleIcon(bond.composite_score || bond.risk_score)}
            >
              <Popup className="custom-popup">
                <div className="popup-content">
                  <div className="popup-title">{(bond.issuer || '').toUpperCase()}</div>

                  <div style={{ margin: '8px 0', padding: '8px', background: '#f9f9f9', borderRadius: '4px' }}>
                    <HazardBar label="Fire" value={bond.wildfire_adjusted || bond.wildfire_score || 0}    color="#e63946" emoji="🔥" />
                    <HazardBar label="Flood" value={bond.flood_score || 0}       color="#457b9d" emoji="🌊" />
                    <HazardBar label="Quake" value={bond.earthquake_score || 0}  color="#6d6875" emoji="🌍" />
                    <HazardBar label="NDVI" value={bond.ndvi || 0}               color="#2d6a4f" emoji="🌿" />
                  </div>

                  <div className="popup-row">
                    <span className="popup-label">Composite Risk</span>
                    <span className="popup-value" style={{ color: '#e63946' }}>
                      {((bond.composite_score || bond.risk_score) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="popup-row">
                    <span className="popup-label">Climate Spread</span>
                    <span className="popup-value">+{(bond.climate_spread_bps || 0).toFixed(0)} bps</span>
                  </div>
                  <div className="popup-row" style={{ borderTop: '1px solid #ddd', marginTop: '4px', paddingTop: '4px' }}>
                    <span className="popup-label">Fair Yield</span>
                    <span className="popup-value">{(bond.fair_value_yield || 0).toFixed(2)}%</span>
                  </div>
                </div>
              </Popup>
            </Marker>
          ))}

          <div className="legend-pill">
            <div className="legend-item">
              <span>🔥 Fire (NDVI-adj) + 🌊 Flood + 🌍 Quake = Composite</span>
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
