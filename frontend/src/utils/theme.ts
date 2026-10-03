export type ThemeChoice = "system" | "light" | "dark";

// Also read by the inline script in index.html, which applies the theme before
// the first paint; keep the two in step.
const STORAGE_KEY = "theme";

const DARK_QUERY = "(prefers-color-scheme: dark)";

/** The saved choice, or "system". Storage can be unavailable or throw. */
export function readThemeChoice(): ThemeChoice {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "light" || saved === "dark" || saved === "system") return saved;
  } catch {
    // Private mode or blocked storage: fall back to following the system.
  }
  return "system";
}

export function saveThemeChoice(choice: ThemeChoice): void {
  try {
    localStorage.setItem(STORAGE_KEY, choice);
  } catch {
    // Not remembered, but still applied for this visit.
  }
}

export function systemPrefersDark(): boolean {
  return window.matchMedia(DARK_QUERY).matches;
}

/** Sets data-theme on <html> to the theme the choice resolves to now. */
export function applyTheme(choice: ThemeChoice): void {
  const dark = choice === "dark" || (choice === "system" && systemPrefersDark());
  document.documentElement.dataset.theme = dark ? "dark" : "light";
}

/** Calls back whenever the system switches between light and dark. */
export function onSystemThemeChange(callback: () => void): () => void {
  const query = window.matchMedia(DARK_QUERY);
  query.addEventListener("change", callback);
  return () => query.removeEventListener("change", callback);
}
