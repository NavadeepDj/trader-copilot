import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { RefreshCw, Loader2, TrendingUp, TrendingDown, Wallet } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";

interface Position {
  symbol: string;
  qty: number;
  entry: number;
  stop: number;
  target: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  opened_at: string;
}

interface EquityPoint {
  timestamp: string;
  portfolio_value: number;
  drawdown_pct: number;
}

interface PortfolioData {
  status: string;
  cash: number;
  total_invested: number;
  portfolio_value: number;
  net_profit_inr: number;
  net_profit_pct: number;
  max_drawdown_pct: number;
  realized_pnl: number;
  unrealized_pnl: number;
  open_positions: Position[];
  recent_equity_curve: EquityPoint[];
  quote_errors?: string[];
}

function formatINR(value: number) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);
}

function formatTime(ts: string) {
  if (!ts) return "";
  return ts.length >= 16 ? ts.slice(5, 16) : ts;
}

export function Portfolio() {
  const [data, setData] = useState<PortfolioData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchPortfolio = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/portfolio");
      const d = await res.json();
      setData(d);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPortfolio();
  }, []);

  const chartData = data?.recent_equity_curve?.map((pt) => ({
    time: formatTime(pt.timestamp),
    value: pt.portfolio_value,
    drawdown: pt.drawdown_pct,
  })) || [];

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-lg font-semibold flex items-center gap-2">
          <Wallet className="h-5 w-5" />
          Portfolio
        </CardTitle>
        <Button variant="ghost" size="icon" onClick={fetchPortfolio} disabled={loading}>
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
          <Tabs defaultValue="overview" className="space-y-4">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="positions">Positions</TabsTrigger>
              <TabsTrigger value="charts">Charts</TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-muted rounded-lg">
                  <p className="text-xs text-muted-foreground">Cash</p>
                  <p className="text-lg font-semibold">{formatINR(data.cash)}</p>
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <p className="text-xs text-muted-foreground">Invested</p>
                  <p className="text-lg font-semibold">{formatINR(data.total_invested)}</p>
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <p className="text-xs text-muted-foreground">Portfolio Value</p>
                  <p className="text-lg font-semibold">{formatINR(data.portfolio_value)}</p>
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <p className="text-xs text-muted-foreground">Net Profit</p>
                  <p className={cn("text-lg font-semibold", data.net_profit_inr >= 0 ? "text-green-600" : "text-red-600")}>
                    {formatINR(data.net_profit_inr)} ({data.net_profit_pct.toFixed(2)}%)
                  </p>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-3 text-sm">
                <div className="flex flex-col">
                  <span className="text-muted-foreground">Max Drawdown</span>
                  <span className="font-medium text-red-600">{data.max_drawdown_pct.toFixed(2)}%</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-muted-foreground">Realized P&L</span>
                  <span className={cn("font-medium", data.realized_pnl >= 0 ? "text-green-600" : "text-red-600")}>
                    {formatINR(data.realized_pnl)}
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-muted-foreground">Unrealized P&L</span>
                  <span className={cn("font-medium", data.unrealized_pnl >= 0 ? "text-green-600" : "text-red-600")}>
                    {formatINR(data.unrealized_pnl)}
                  </span>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="positions" className="space-y-2">
              {data.open_positions.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-4">No open positions</p>
              ) : (
                data.open_positions.map((pos, i) => (
                  <div key={i} className="p-3 bg-muted rounded-lg space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold">{pos.symbol.replace(".NS", "")}</span>
                      <Badge variant={pos.unrealized_pnl >= 0 ? "default" : "destructive"}>
                        {pos.unrealized_pnl >= 0 ? <TrendingUp className="h-3 w-3 mr-1" /> : <TrendingDown className="h-3 w-3 mr-1" />}
                        {formatINR(pos.unrealized_pnl)}
                      </Badge>
                    </div>
                    <div className="grid grid-cols-4 gap-2 text-xs">
                      <div>
                        <span className="text-muted-foreground">Qty</span>
                        <p className="font-medium">{pos.qty}</p>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Entry</span>
                        <p className="font-medium">₹{pos.entry.toFixed(2)}</p>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Stop</span>
                        <p className="font-medium text-red-600">₹{pos.stop.toFixed(2)}</p>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Target</span>
                        <p className="font-medium text-green-600">₹{pos.target.toFixed(2)}</p>
                      </div>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Current: ₹{pos.current_price.toFixed(2)} | Value: {formatINR(pos.market_value)}
                    </div>
                  </div>
                ))
              )}
            </TabsContent>

            <TabsContent value="charts" className="space-y-4">
              {chartData.length < 2 ? (
                <p className="text-sm text-muted-foreground text-center py-4">
                  Need at least 2 equity points for charts
                </p>
              ) : (
                <>
                  <div>
                    <p className="text-sm font-medium mb-2">Equity Curve</p>
                    <div className="h-32">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData}>
                          <defs>
                            <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="hsl(var(--chart-1))" stopOpacity={0.3} />
                              <stop offset="95%" stopColor="hsl(var(--chart-1))" stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                          <YAxis tick={{ fontSize: 10 }} domain={["auto", "auto"]} />
                          <Tooltip formatter={(v) => [formatINR(Number(v)), "Value"]} />
                          <Area
                            type="monotone"
                            dataKey="value"
                            stroke="hsl(var(--chart-1))"
                            fill="url(#colorValue)"
                            strokeWidth={2}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  <div>
                    <p className="text-sm font-medium mb-2">Drawdown %</p>
                    <div className="h-32">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData}>
                          <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                          <YAxis tick={{ fontSize: 10 }} domain={["auto", 0]} />
                          <Tooltip formatter={(v) => [`${Number(v).toFixed(2)}%`, "Drawdown"]} />
                          <Line
                            type="monotone"
                            dataKey="drawdown"
                            stroke="hsl(var(--destructive))"
                            strokeWidth={2}
                            dot={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </>
              )}
            </TabsContent>
          </Tabs>
        ) : null}
      </CardContent>
    </Card>
  );
}
