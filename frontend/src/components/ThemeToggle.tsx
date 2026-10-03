import { useEffect, useState } from "react";
import {
  applyTheme,
  onSystemThemeChange,
  readThemeChoice,
  saveThemeChoice,
  type ThemeChoice,
} from "../utils/theme";

const OPTIONS: { value: ThemeChoice; icon: string; label: string }[] = [
  { value: "system", icon: "◐", label: "System theme" },
  { value: "light", icon: "☀", label: "Light theme" },
  { value: "dark", icon: "☾", label: "Dark theme" },
];

/** System / light / dark, remembered across visits. */
export function ThemeToggle() {
  const [choice, setChoice] = useState<ThemeChoice>(readThemeChoice);

  useEffect(() => {
    applyTheme(choice);
    saveThemeChoice(choice);
    // Following the system means following it while the page is open, too.
    if (choice !== "system") return;
    return onSystemThemeChange(() => applyTheme("system"));
  }, [choice]);

  return (
    <div
      role="radiogroup"
      aria-label="Theme"
      style={{
        position: "fixed",
        top: 16,
        right: 16,
        display: "flex",
        gap: 2,
        padding: 3,
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
        borderRadius: 999,
        boxShadow: "var(--shadow-sm)",
        zIndex: 20,
      }}
    >
      {OPTIONS.map((option) => {
        const selected = option.value === choice;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={selected}
            aria-label={option.label}
            title={option.label}
            onClick={() => setChoice(option.value)}
            style={{
              width: 28,
              height: 28,
              border: "none",
              borderRadius: 999,
              background: selected ? "var(--color-accent-bg)" : "transparent",
              color: selected ? "var(--color-accent)" : "var(--color-text-muted)",
              fontSize: 14,
              lineHeight: 1,
              cursor: "pointer",
              transition: "all 0.12s",
            }}
          >
            {option.icon}
          </button>
        );
      })}
    </div>
  );
}
