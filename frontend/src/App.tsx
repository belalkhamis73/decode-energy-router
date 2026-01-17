import React, { useState, useEffect } from 'react';
import RealTimeGrid, { GridNode, GridLink } from './components/RealTimeGrid';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, Zap, AlertTriangle, Play, RefreshCw, Server } from 'lucide-react';

// --- Types & Interfaces ---
interface TelemetryPoint {
  time: string;
  voltage: number;
  frequency: number;
}

const App: React.FC = () => {
  // --- State Management ---
  const [nodes, setNodes] = useState<GridNode[]>([]);
  const [links, setLinks] = useState<GridLink[]>([]);
  const [telemetry, setTelemetry] = useState<TelemetryPoint[]>([]);
  const [systemStatus, setSystemStatus] = useState<'healthy' | 'critical' | 'simulating'>('healthy');
  const [lastUpdate, setLastUpdate] = useState<string>(new Date().toISOString());

  // --- Mock Data Generator (Simulates Backend/WebSocket) ---
  const fetchGridState = () => {
    // In production, this calls: axios.get('/api/v1/simulation/latest')
    
    // Simulate dynamic voltage fluctuation
    const noise = () => (Math.random() - 0.5) * 0.04;
    
    const mockNodes: GridNode[] = [
      { id: "Bus_1", type: "gen", voltage_pu: 1.0 + noise(), x: 100, y: 300 },
      { id: "Bus_2", type: "bus", voltage_pu: 0.98 + noise(), x: 300, y: 150 },
      { id: "Bus_3", type: "load", voltage_pu: 0.96 + noise(), x: 300, y: 450 },
      { id: "Bus_4", type: "bus", voltage_pu: 0.97 + noise(), x: 500, y: 300 },
      { id: "Bus_5", type: "gen", voltage_pu: 1.02 + noise(), x: 700, y: 300 },
    ];

    const mockLinks: GridLink[] = [
      { source: "Bus_1", target: "Bus_2", loading_pct: 45 + Math.random() * 10, status: 'closed' },
      { source: "Bus_1", target: "Bus_3", loading_pct: 60 + Math.random() * 10, status: 'closed' },
      { source: "Bus_2", target: "Bus_4", loading_pct: 30, status: 'closed' },
      { source: "Bus_3", target: "Bus_4", loading_pct: 35, status: 'closed' },
      { source: "Bus_4", target: "Bus_5", loading_pct: 20, status: 'closed' },
    ];

    setNodes(mockNodes);
    setLinks(mockLinks);
    setLastUpdate(new Date().toLocaleTimeString());

    // Update Telemetry Chart
    const newPoint: TelemetryPoint = {
      time: new Date().toLocaleTimeString(),
      voltage: 0.98 + noise(),
      frequency: 60.0 + (Math.random() - 0.5) * 0.1
    };
    
    setTelemetry(prev => [...prev.slice(-19), newPoint]); // Keep last 20 points
  };

  // --- Lifecycle Loops ---
  useEffect(() => {
    // Polling loop (Simulates 1Hz SCADA update)
    const interval = setInterval(fetchGridState, 1000);
    return () => clearInterval(interval);
  }, []);

  // --- Handlers ---
  const handleSimulationTrigger = async () => {
    setSystemStatus('simulating');
    // Call Backend: POST /api/v1/simulation/run
    console.log("Triggering Physics Simulation...");
    
    setTimeout(() => {
      setSystemStatus('healthy'); // Reset after "sim"
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-indigo-500/30">
      
      {/* --- Top Navigation --- */}
      <nav className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="bg-indigo-600 p-2 rounded-lg">
                <Activity className="h-6 w-6 text-white" />
              </div>
              <span className="text-xl font-bold tracking-tight">D.E.C.O.D.E <span className="text-indigo-400">Energy Router</span></span>
            </div>
            
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800 border border-slate-700">
                <div className={`w-2 h-2 rounded-full ${systemStatus === 'healthy' ? 'bg-green-500 animate-pulse' : 'bg-yellow-500'}`} />
                <span className="uppercase tracking-wider font-semibold text-xs">{systemStatus}</span>
              </div>
              <div className="flex items-center gap-2 text-slate-400">
                <Server className="h-4 w-4" />
                <span>Edge-01 (NVIDIA Jetson)</span>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* --- Main Dashboard Content --- */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* KPI Headers */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <KpiCard title="System Frequency" value={`${telemetry[telemetry.length - 1]?.frequency.toFixed(3) || '60.00'} Hz`} icon={<Activity className="text-blue-400" />} />
          <KpiCard title="Avg Voltage" value={`${telemetry[telemetry.length - 1]?.voltage.toFixed(3) || '1.00'} p.u.`} icon={<Zap className="text-yellow-400" />} />
          <KpiCard title="Active Faults" value="0" icon={<AlertTriangle className="text-green-400" />} />
          <KpiCard title="Physics Residual" value="1.2e-5" icon={<RefreshCw className="text-purple-400" />} subtitle="Converged" />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Left Column: Topology Visualizer */}
          <div className="lg:col-span-2 space-y-6">
            <RealTimeGrid nodes={nodes} links={links} width={800} height={500} />
            
            {/* Control Deck */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-white">Simulation Controls</h3>
                <p className="text-slate-400 text-sm">Inject faults to test PINN stability response.</p>
              </div>
              <div className="flex gap-4">
                <button 
                  onClick={fetchGridState}
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg transition-colors flex items-center gap-2"
                >
                  <RefreshCw className="h-4 w-4" /> Refresh
                </button>
                <button 
                  onClick={handleSimulationTrigger}
                  className="px-6 py-2 bg-indigo-600 hover:bg-indigo-500 text-white font-medium rounded-lg transition-all shadow-lg shadow-indigo-500/20 flex items-center gap-2"
                >
                  <Play className="h-4 w-4" /> Run Contingency
                </button>
              </div>
            </div>
          </div>

          {/* Right Column: Telemetry Charts */}
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
              <h3 className="text-slate-200 font-semibold mb-4">Frequency Stability</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={telemetry}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="time" hide />
                    <YAxis domain={[59.8, 60.2]} stroke="#94a3b8" fontSize={12} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px' }}
                      itemStyle={{ color: '#fff' }}
                    />
                    <Line type="monotone" dataKey="frequency" stroke="#60a5fa" strokeWidth={2} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
              <h3 className="text-slate-200 font-semibold mb-4">Voltage Profile (Bus 1)</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={telemetry}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="time" hide />
                    <YAxis domain={[0.9, 1.1]} stroke="#94a3b8" fontSize={12} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px' }}
                      itemStyle={{ color: '#fff' }}
                    />
                    <Line type="monotone" dataKey="voltage" stroke="#facc15" strokeWidth={2} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

        </div>
      </main>
    </div>
  );
};

// --- Subcomponents ---
const KpiCard: React.FC<{ title: string, value: string, icon: React.ReactNode, subtitle?: string }> = ({ title, value, icon, subtitle }) => (
  <div className="bg-slate-900 border border-slate-800 p-5 rounded-xl flex items-center justify-between hover:border-slate-700 transition-colors">
    <div>
      <p className="text-slate-400 text-sm font-medium">{title}</p>
      <div className="flex items-baseline gap-2 mt-1">
        <h4 className="text-2xl font-bold text-white">{value}</h4>
        {subtitle && <span className="text-xs text-green-400 font-mono">{subtitle}</span>}
      </div>
    </div>
    <div className="p-3 bg-slate-800/50 rounded-lg">
      {icon}
    </div>
  </div>
);

export default App;
       
