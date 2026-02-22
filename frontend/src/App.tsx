import { useState, useEffect } from "react";
import { Chat } from "@/components/Chat";
import { MarketRegime } from "@/components/MarketRegime";
import { Portfolio } from "@/components/Portfolio";
import { SignalBoard } from "@/components/SignalBoard";
import { Button } from "@/components/ui/button";
import { Moon, Sun, TrendingUp } from "lucide-react";

function App() {
  const [darkMode, setDarkMode] = useState(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("theme") === "dark" ||
        (!localStorage.getItem("theme") && window.matchMedia("(prefers-color-scheme: dark)").matches);
    }
    return true;
  });

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [darkMode]);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-14 items-center justify-between px-4">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-6 w-6 text-primary" />
            <h1 className="text-xl font-bold">Agentic Trading Assistant</h1>
            <span className="ml-3 px-2 py-0.5 text-xs font-medium bg-yellow-500/10 text-yellow-600 rounded-full border border-yellow-500/20">
              PAPER TRADING
            </span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setDarkMode(!darkMode)}
          >
            {darkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
          </Button>
        </div>
      </header>

      {/* Main Layout */}
      <main className="container px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Chat Panel - Takes 1 column */}
          <div className="lg:col-span-1 h-[calc(100vh-120px)]">
            <Chat />
          </div>

          {/* Dashboard Panel - Takes 2 columns */}
          <div className="lg:col-span-2 space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <MarketRegime />
              <Portfolio />
            </div>
            <SignalBoard />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
