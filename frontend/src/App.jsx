import { useState, useEffect } from 'react'
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet'
import axios from 'axios'
import 'leaflet/dist/leaflet.css'
import './index.css'

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

  const getRiskColor = (score) => {
    if (score > 0.7) return '#ef4444' // Red
    if (score > 0.4) return '#f59e0b' // Orange
    return '#10b981' // Green
  }

  // Calculate center of map (CA)
  const mapCenter = [37.0, -119.0]

  if (loading) return <div className="loading">Loading Analytics Engine...</div>

  return (
    <div className="app-container">
      <div className="sidebar">
        <h1 className="title">ICE Climate Risk</h1>
        <p className="subtitle">Municipal Bond Portfolio Analytics</p>

        {stats && (
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-label">Total Exposure</div>
              <div className="stat-value">${(stats.total_exposure / 1e9).toFixed(2)}B</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Climate VaR (Wildfire)</div>
              <div className="stat-value" style={{ color: '#ef4444' }}>
                ${(stats.total_var / 1e6).toFixed(1)}M
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Avg Risk Score</div>
              <div className="stat-value">{(stats.avg_risk_score * 100).toFixed(1)}/100</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">High Risk Bonds</div>
              <div className="stat-value">{stats.high_risk_bonds}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Avg Climate Spread</div>
              <div className="stat-value" style={{ color: '#f59e0b' }}>+{(stats.avg_spread_bps || 0).toFixed(0)} bps</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Portfolio Size</div>
              <div className="stat-value">{stats.num_bonds || 0} Bonds</div>
            </div>
          </div>
        )}

        <div style={{ marginTop: 'auto', fontSize: '0.8rem', color: '#666' }}>
          &copy; 2026 Climate Analytics Prototype
        </div>
      </div>

      <div className="map-container">
        <MapContainer center={mapCenter} zoom={6} scrollWheelZoom={true} style={{ height: "100%", width: "100%" }}>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          />
          {bonds.map((bond) => (
            <CircleMarker
              key={bond.bond_id}
              center={[bond.lat, bond.lon]}
              pathOptions={{
                color: getRiskColor(bond.risk_score),
                fillColor: getRiskColor(bond.risk_score),
                fillOpacity: 0.7,
                weight: 1
              }}
              radius={bond.risk_score * 15 + 3} // Size relative to risk
            >
              <Popup className="custom-popup">
                <div style={{ color: '#0f172a' }}>
                  <strong>{bond.issuer}</strong> <span style={{ fontSize: '0.8em', color: '#64748b' }}>({bond.bond_id})</span><br />
                  <div style={{ margin: '4px 0', padding: '4px', background: '#f1f5f9', borderRadius: '4px' }}>
                    Rating: <strong>{bond.rating || 'N/A'}</strong> | Yld: {(bond.coupon_rate || 0).toFixed(2)}%
                  </div>
                  Risk Score: {(bond.risk_score || 0).toFixed(2)}<br />
                  Climate Spread: <span style={{ color: '#ef4444' }}>+{(bond.climate_spread_bps || 0).toFixed(0)} bps</span><br />
                  Fair Value: <strong>{(bond.fair_value_yield || 0).toFixed(2)}%</strong><br />
                  VaR: ${((bond.VaR_Amount || 0) / 1e6).toFixed(2)}M
                </div>
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>
    </div>
  )
}

export default App
