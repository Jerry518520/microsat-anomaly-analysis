declare module 'plotly.js-dist' {
  const Plotly: {
    newPlot(div: string | HTMLElement, data: unknown[], layout?: unknown, config?: unknown): Promise<HTMLElement>;
    purge(div: string | HTMLElement): void;
    Plots: {
      resize(div: string | HTMLElement): void;
    };
  };
  export default Plotly;
}
