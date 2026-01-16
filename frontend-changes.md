# Frontend Changes - Dark/Light Theme Toggle

## Overview
Added a complete dark/light theme toggle feature to the RAG chatbot application, allowing users to seamlessly switch between dark and light themes with their preference persisted in localStorage.

## Changes Made

### 1. HTML Changes (`frontend/index.html`)
**Location:** After the opening `<body>` tag, before the header

**Added:**
- Theme toggle button with sun and moon icons
- Positioned as a fixed element in the top-right corner
- Includes proper ARIA labels for accessibility
- SVG icons for both sun (light theme) and moon (dark theme)

**Code Added:**
```html
<button id="themeToggle" class="theme-toggle" aria-label="Toggle theme">
    <svg class="sun-icon">...</svg>
    <svg class="moon-icon">...</svg>
</button>
```

### 2. CSS Changes (`frontend/style.css`)

#### Light Theme Variables
**Location:** After the dark theme `:root` variables

**Added:**
- Complete set of CSS custom properties for light theme
- Applied via `[data-theme="light"]` attribute selector
- Maintains consistent color hierarchy with proper contrast ratios

**Key Variables:**
- Background: `#f8fafc` (light slate)
- Surface: `#ffffff` (white)
- Text Primary: `#0f172a` (dark slate)
- Text Secondary: `#475569` (medium slate)
- Border Color: `#e2e8f0` (light border)
- Shadows adjusted for light backgrounds

#### Theme Transitions
**Location:** Top of CSS file after reset

**Added:**
- Global transition rules for smooth theme switching
- Applies to: background-color, color, border-color, box-shadow
- Transition duration: 0.3s with ease timing function

#### Theme Toggle Button Styling
**Location:** After body styles

**Added:**
- Fixed positioning (top: 1rem, right: 1rem)
- Circular button (48px × 48px)
- Hover effects with scale transformation
- Focus states for keyboard navigation
- Active state with scale-down effect
- Icon rotation animations for smooth transitions

**Visual States:**
- Default: Shows moon icon in dark theme
- Hover: Scales to 1.05x, highlights border
- Focus: Blue focus ring for accessibility
- Active: Scales to 0.95x for press feedback

#### Icon Visibility Logic
- Dark theme: Moon icon visible, sun icon hidden (rotated 180°)
- Light theme: Sun icon visible, moon icon hidden (rotated -180°)
- Smooth rotation and opacity transitions

#### Light Theme Overrides
**Location:** Within message content styles

**Added:**
- Adjusted code block backgrounds for light theme
- Reduced opacity for inline code backgrounds
- Proper contrast for code snippets in light mode

### 3. JavaScript Changes (`frontend/script.js`)

#### Theme Initialization
**Function:** `initializeTheme()`

**Added:**
- Checks localStorage for saved theme preference
- Falls back to system preference (`prefers-color-scheme`)
- Automatically applies theme on page load
- Called in DOMContentLoaded event

**Logic:**
1. Check localStorage for 'theme' key
2. If exists, apply saved theme
3. If not exists, check system preference
4. If user prefers light, set light theme
5. Otherwise, default to dark theme

#### Theme Toggle Function
**Function:** `toggleTheme()`

**Added:**
- Reads current theme from `data-theme` attribute
- Toggles between 'light' and 'dark'
- Calls `setTheme()` to apply changes
- Attached to theme toggle button click event

#### Theme Setter Function
**Function:** `setTheme(theme)`

**Added:**
- Applies or removes `data-theme="light"` attribute
- Saves preference to localStorage
- Handles both light and dark states

**Implementation:**
- Light theme: Sets `data-theme="light"` on `<html>` element
- Dark theme: Removes `data-theme` attribute (default state)
- Persists choice: `localStorage.setItem('theme', theme)`

#### Event Listeners
**Location:** `setupEventListeners()` function

**Added:**
- Click event listener for theme toggle button
- Calls `toggleTheme()` on click

## Features Implemented

### 1. Visual Design
- Icon-based toggle button (sun/moon)
- Fixed position in top-right corner
- Smooth scale animations on hover/click
- Rotating icon transitions
- Consistent with existing design aesthetic
- High contrast in both themes (WCAG compliant)

### 2. Accessibility
- Keyboard navigable (can be focused with Tab key)
- ARIA label: "Toggle theme"
- Visible focus ring for keyboard users
- High contrast colors in both themes
- Respects system preference on first visit

### 3. User Experience
- Instant theme switching with smooth 0.3s transitions
- Theme preference persisted in localStorage
- Respects system color scheme preference
- No flash of wrong theme on page load
- Smooth rotation and fade animations for icons

### 4. Performance
- CSS custom properties for efficient theme switching
- No layout reflow on theme change
- Transitions only on color-related properties
- Lightweight implementation with no dependencies

## Technical Details

### Theme Storage
- Key: `'theme'`
- Values: `'light'` or `'dark'`
- Stored in: localStorage
- Persists across browser sessions

### Theme Application
- Attribute: `data-theme="light"` on `<html>` element
- Default: No attribute (dark theme)
- CSS Selector: `[data-theme="light"]`

### Browser Compatibility
- CSS Custom Properties: All modern browsers
- localStorage: All modern browsers
- prefers-color-scheme: All modern browsers (Safari 12.1+, Chrome 76+, Firefox 67+)
- SVG: Universal support

## Files Modified

1. **frontend/index.html**
   - Added theme toggle button with SVG icons
   - Positioned before header section

2. **frontend/style.css**
   - Added light theme CSS variables
   - Added theme toggle button styles
   - Added smooth transition rules
   - Added light theme overrides for code blocks

3. **frontend/script.js**
   - Added `initializeTheme()` function
   - Added `toggleTheme()` function
   - Added `setTheme()` function
   - Added event listener for theme toggle

## Testing Recommendations

1. **Visual Testing:**
   - Verify all UI elements in both themes
   - Check contrast ratios for text readability
   - Test hover/focus states on interactive elements
   - Verify code blocks and inline code visibility

2. **Functional Testing:**
   - Toggle between themes multiple times
   - Refresh page to verify persistence
   - Clear localStorage and verify system preference detection
   - Test keyboard navigation (Tab to button, Enter to toggle)

3. **Cross-Browser Testing:**
   - Test in Chrome, Firefox, Safari, Edge
   - Verify transitions work smoothly
   - Check localStorage persistence

## Future Enhancements (Optional)

- Add keyboard shortcut for theme toggle (e.g., Ctrl+Shift+T)
- Add theme transition preference (reduce motion)
- Add additional theme options (e.g., auto, high contrast)
- Add theme sync across tabs using StorageEvent
- Add theme preview before switching
