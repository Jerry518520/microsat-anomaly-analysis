import { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist';

interface PlotProps {
  data: Record<string, unknown>[];
  layout: Record<string, unknown>;
  config?: Record<string, unknown>;
  style?: React.CSSProperties;
  className?: string;
}

/** Plotly 封装 — 数据变化用 react 增量更新,容器尺寸变化自动 resize */
export default function Plot({ data, layout, config, style, className }: PlotProps) {
  const ref = useRef<HTMLDivElement>(null);
  const mounted = useRef(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (!mounted.current) {
      Plotly.newPlot(el, data, layout, config);
      mounted.current = true;
    } else {
      Plotly.react(el, data, layout, config);
    }
  }, [data, layout, config]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver(() => Plotly.Plots.resize(el));
    observer.observe(el);
    return () => {
      observer.disconnect();
      Plotly.purge(el);
      mounted.current = false;
    };
  }, []);

  return <div ref={ref} style={style} className={className} />;
}
