import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { RefreshCw, TrendingUp, TrendingDown, Minus, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface RegimeData {
  status: string;
  regime: string;
  strategy: string;
  metrics: {
    close: number;
    dma_50: number;
    dma_50_slope: number;
    return_20d: number;
    volatility: number;
  };
  source: string;
  fetched_at_ist: string;
  last_trade_date: string;
  error_message?: string;
}

export function MarketRegime() {
  const [data, setData] = useState<RegimeData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchRegime = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/regime");
      const d = await res.json();
      if (d.status !== "success") {
        setError(d.error_message || "Failed to fetch regime");
      } else {
        setData(d);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRegime();
  }, []);

  const getRegimeIcon = (regime: string) => {
    if (regime?.toLowerCase().includes("bull")) return <TrendingUp className="h-4 w-4" />;
    if (regime?.toLowerCase().includes("bear")) return <TrendingDown className="h-4 w-4" />;
    return <Minus className="h-4 w-4" />;
  };

  const getRegimeColor = (regime: string) => {
    if (regime?.toLowerCase().includes("bull")) return "bg-green-500/10 text-green-600 border-green-500/20";
    if (regime?.toLowerCase().includes("bear")) return "bg-red-500/10 text-red-600 border-red-500/20";
    return "bg-yellow-500/10 text-yellow-600 border-yellow-500/20";
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-lg font-semibold">Market Regime</CardTitle>
        <Button variant="ghost" size="icon" onClick={fetchRegime} disabled={loading}>
          <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
        </Button>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : error ? (
          <p className="text-sm text-destructive">{error}</p>
        ) : data ? (
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <Badge
                variant="outline"
                className={cn("text-lg px-4 py-1 font-semibold", getRegimeColor(data.regime))}
              >
                {getRegimeIcon(data.regime)}
                <span className="ml-2">{data.regime}</span>
              </Badge>
            </div>
            <p className="text-sm text-muted-foreground">
              Strategy: <span className="font-medium text-foreground">{data.strategy}</span>
            </p>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Nifty Close</span>
                <span className="font-medium">{data.metrics.close?.toLocaleString("en-IN")}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">50-DMA</span>
                <span className="font-medium">{data.metrics.dma_50?.toLocaleString("en-IN")}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">DMA Slope</span>
                <span className="font-medium">{data.metrics.dma_50_slope?.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">20d Return</span>
                <span className={cn("font-medium", (data.metrics.return_20d ?? 0) >= 0 ? "text-green-600" : "text-red-600")}>
                  {((data.metrics.return_20d ?? 0) * 100).toFixed(2)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Volatility</span>
                <span className="font-medium">{((data.metrics.volatility ?? 0) * 100).toFixed(2)}%</span>
              </div>
            </div>
            <p className="text-xs text-muted-foreground pt-2 border-t">
              Source: {data.source} | {data.fetched_at_ist}
              <br />
              Last trade: {data.last_trade_date}
            </p>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
