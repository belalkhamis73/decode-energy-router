import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'; // Assuming shadcn/ui or similar

// --- Data Contracts (Strict Typing) ---
export interface GridNode {
  id: string;
  type: 'gen' | 'load' | 'bus';
  voltage_pu: number;
  x?: number; // Optional fixed coordinates
  y?: number;
}

export interface GridLink {
  source: string;
  target: string;
  loading_pct: number; // 0 to 100+
  status: 'closed' | 'tripped';
}

interface RealTimeGridProps {
  nodes: GridNode[];
  links: GridLink[];
  width?: number;
  height?: number;
}

const RealTimeGrid: React.FC<RealTimeGridProps> = ({ nodes, links, width = 800, height = 600 }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null);

  // --- Physics-Based Color Scales ---
  // Voltage: Green at 1.0, Red at <0.95 or >1.05
  const getVoltageColor = (v: number) => {
    if (v < 0.95 || v > 1.05) return "#ef4444"; // Red-500 (Critical)
    if (v < 0.98 || v > 1.02) return "#eab308"; // Yellow-500 (Warning)
    return "#22c55e"; // Green-500 (Nominal)
  };

  // Line Loading: Thicker and Redder as load increases
  const getLinkColor = (pct: number) => {
    return pct > 90 ? "#ef4444" : "#94a3b8"; // Red if overloaded, else Slate-400
  };

  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    // 1. Setup D3 Context
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // Clear previous render (Simple cleanup)

    // 2. Define Simulation (Physics-based Layout)
    // Uses repulsion (Charge) and springs (Links) to organize the topology
    const simulation = d3.forceSimulation(nodes as d3.SimulationNodeDatum[])
      .force("link", d3.forceLink(links).id((d: any) => d.id).distance(100))
      .force("charge", d3.forceManyBody().strength(-300)) // Repel nodes
      .force("center", d3.forceCenter(width / 2, height / 2))
      .stop(); // We will manually tick or let it run

    // 3. Render Links (Transmission Lines)
    const linkGroup = svg.append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(links)
      .enter().append("line")
      .attr("stroke-width", d => Math.max(1, (d.loading_pct / 100) * 4)) // Thickness = Load
      .attr("stroke", d => d.status === 'tripped' ? "#99f6e4" : getLinkColor(d.loading_pct))
      .attr("stroke-dasharray", d => d.status === 'tripped' ? "5,5" : "none") // Dashed if tripped
      .attr("opacity", 0.7);

    // 4. Render Nodes (Buses/Generators)
    const nodeGroup = svg.append("g")
      .attr("class", "nodes")
      .selectAll("circle")
      .data(nodes)
      .enter().append("circle")
      .attr("r", d => d.type === 'gen' ? 12 : 8) // Generators are larger
      .attr("fill", d => getVoltageColor(d.voltage_pu))
      .attr("stroke", "#1e293b") // Slate-800 border
      .attr("stroke-width", 2)
      .call(d3.drag<SVGCircleElement, any>() // Allow operators to rearrange grid
        .on("start", dragStarted)
        .on("drag", dragged)
        .on("end", dragEnded)
      );

    // 5. Add Labels
    const labelGroup = svg.append("g")
      .selectAll("text")
      .data(nodes)
      .enter().append("text")
      .text(d => d.id)
      .attr("font-size", 10)
      .attr("dx", 15)
      .attr("dy", 4)
      .attr("fill", "#cbd5e1"); // Slate-300

    // 6. Interaction Logic (Tooltips)
    nodeGroup.on("mouseover", (event, d) => {
      setTooltip({
        x: event.pageX,
        y: event.pageY,
        content: `Bus: ${d.id} | Voltage: ${d.voltage_pu.toFixed(3)} p.u.`
      });
    })
    .on("mouseout", () => setTooltip(null));

    // 7. Simulation Loop
    simulation.on("tick", () => {
      linkGroup
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y);

      nodeGroup
        .attr("cx", (d: any) => d.x)
        .attr("cy", (d: any) => d.y);

      labelGroup
        .attr("x", (d: any) => d.x)
        .attr("y", (d: any) => d.y);
    });

    simulation.restart();

    // --- Drag Helpers ---
    function dragStarted(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event: any, d: any) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragEnded(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    // Cleanup
    return () => {
      simulation.stop();
    };
  }, [nodes, links, width, height]); // Re-render when physical state changes

  return (
    <Card className="w-full bg-slate-950 border-slate-800 text-slate-200">
      <CardHeader>
        <CardTitle className="flex justify-between items-center">
          <span>Live Topology</span>
          <div className="flex gap-2 text-xs">
            <span className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-green-500"></div> Normal</span>
            <span className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-red-500"></div> Critical</span>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="relative">
        <svg 
          ref={svgRef} 
          width={width} 
          height={height} 
          className="bg-slate-900 rounded-lg shadow-inner border border-slate-800"
          style={{ cursor: 'grab' }}
        />
        
        {/* React Portal or Absolute Div for Tooltip */}
        {tooltip && (
          <div 
            className="absolute z-50 bg-slate-800 border border-slate-700 p-2 rounded shadow-lg text-xs pointer-events-none"
            style={{ left: tooltip.x - 20, top: tooltip.y - 40 }} // Adjust offset relative to container if needed
          >
            {tooltip.content}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default RealTimeGrid;
