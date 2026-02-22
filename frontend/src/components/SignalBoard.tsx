import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { RefreshCw, Loader2, BarChart3 } from "lucide-react";
import { cn } from "@/lib/utils";

interface Signal {
  symbol: string;
  display_symbol?: string;
  signal: "BUY" | "SELL" | "HOLD";
  current_price: number;
  entry: number;
  stop: number;
  target: number;
  rationale: string;
  metrics?: {
    rsi?: number;
  };
  news?: Array<{ title: string; publisher: string }>;
  news_today_count?: number;
  news_recent_count?: number;
  news_error?: string;
}

interface SignalBoardData {
  status: string;
  regime: string;
  strategy: string;
  signal_counts: {
    BUY?: number;
    HOLD?: number;
    SELL?: number;
  };
  signals: Signal[];
  stocks_scanned: number;
  stocks_requested: number;
  generated_at_ist: string;
  source: string;
  scan_errors?: string[];
  error_message?: string;
}

function formatINR(value: number | null | undefined) {
  if (value === null || value === undefined || isNaN(value)) return "-";
  return value.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export function SignalBoard() {
  const [data, setData] = useState<SignalBoardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSignals = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/signals/nifty50?include_news=true&max_news=2&news_days=1");
      const d = await res.json();
      if (d.status !== "success") {
        setError(d.error_message || "Failed to fetch signals");
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
    fetchSignals();
  }, []);

  const getSignalBadge = (signal: string) => {
    switch (signal) {
      case "BUY":
        return <Badge className="bg-green-500/10 text-green-600 border-green-500/20 hover:bg-green-500/20">BUY</Badge>;
      case "SELL":
        return <Badge className="bg-red-500/10 text-red-600 border-red-500/20 hover:bg-red-500/20">SELL</Badge>;
      default:
        return <Badge variant="secondary">HOLD</Badge>;
    }
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-lg font-semibold flex items-center gap-2">
          <BarChart3 className="h-5 w-5" />
          Nifty 50 Signal Board
        </CardTitle>
        <Button variant="ghost" size="icon" onClick={fetchSignals} disabled={loading}>
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
            {/* Summary */}
            <div className="grid grid-cols-4 gap-3 text-sm">
              <div className="flex flex-col">
                <span className="text-muted-foreground">Regime</span>
                <span className="font-medium">{data.regime || "-"}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-muted-foreground">Strategy</span>
                <span className="font-medium">{data.strategy || "-"}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-muted-foreground">Signals</span>
                <div className="flex gap-1 text-xs">
                  <span className="text-green-600 font-medium">{data.signal_counts?.BUY || 0} BUY</span>
                  <span>|</span>
                  <span className="text-muted-foreground">{data.signal_counts?.HOLD || 0} HOLD</span>
                  <span>|</span>
                  <span className="text-red-600 font-medium">{data.signal_counts?.SELL || 0} SELL</span>
                </div>
              </div>
              <div className="flex flex-col">
                <span className="text-muted-foreground">Scanned</span>
                <span className="font-medium">{data.stocks_scanned}/{data.stocks_requested}</span>
              </div>
            </div>

            {/* Table */}
            <ScrollArea className="h-[400px]">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[100px]">Symbol</TableHead>
                    <TableHead>Signal</TableHead>
                    <TableHead className="text-right">Price</TableHead>
                    <TableHead className="text-right">Entry</TableHead>
                    <TableHead className="text-right">Stop</TableHead>
                    <TableHead className="text-right">Target</TableHead>
                    <TableHead className="text-right">RSI</TableHead>
                    <TableHead className="w-[200px]">Rationale</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.signals.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={8} className="text-center text-muted-foreground">
                        No signals available
                      </TableCell>
                    </TableRow>
                  ) : (
                    data.signals.map((signal, i) => (
                      <TableRow key={i}>
                        <TableCell className="font-medium">
                          {(signal.display_symbol || signal.symbol).replace(".NS", "")}
                        </TableCell>
                        <TableCell>{getSignalBadge(signal.signal)}</TableCell>
                        <TableCell className="text-right">₹{formatINR(signal.current_price)}</TableCell>
                        <TableCell className="text-right">₹{formatINR(signal.entry)}</TableCell>
                        <TableCell className="text-right text-red-600">₹{formatINR(signal.stop)}</TableCell>
                        <TableCell className="text-right text-green-600">₹{formatINR(signal.target)}</TableCell>
                        <TableCell className="text-right">
                          {signal.metrics?.rsi != null ? signal.metrics.rsi.toFixed(1) : "-"}
                        </TableCell>
                        <TableCell className="text-xs text-muted-foreground max-w-[200px] truncate">
                          {signal.rationale || "-"}
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </ScrollArea>

            {data.scan_errors && data.scan_errors.length > 0 && (
              <p className="text-xs text-destructive">
                Scan errors: {data.scan_errors.join(" | ")}
              </p>
            )}

            <p className="text-xs text-muted-foreground border-t pt-2">
              Generated: {data.generated_at_ist} | Source: {data.source}
            </p>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
