# CodaCite Design System

CodaCite uses a vibrant, high-contrast design system built on Vanilla CSS with glassmorphism principles and HSL-based color tokens.

## Design Tokens

### Colors (HSL)

- **Primary**: `hsl(245, 85%, 65%)` (Vibrant Indigo)
- **Secondary**: `hsl(320, 80%, 60%)` (Electric Pink)
- **Accent**: `hsl(160, 80%, 50%)` (Emerald Green)
- **Background**: `hsl(222, 47%, 7%)` (Deep Space Blue)

### Typography

- **Headings**: [Outfit](https://fonts.google.com/specimen/Outfit) (800 weight)
- **Body**: [Inter](https://fonts.google.com/specimen/Inter) (400-700 weight)
- **Base Font Size**: `125%` (approx 20px) for maximum readability on high-resolution displays.

## Layout Components

### Sidebar (`aside`)

- **Width**: 420px
- **Role**: Navigation, Notebook selection, and Source management.
- **Visuals**: Solid background (`--bg-sidebar`) with a subtle right border.

### Main Content (`main`)

- **Role**: Chat interface and Search results.
- **Visuals**: Radial gradients and glassmorphism cards.

### Chat Messages (`.message`)

- **User**: Gradient background with indigo glow.
- **Assistant**: Glassy card with heavy blur and subtle border.

## Animations

- **Entrance**: `fadeInSlideUp` (0.6s cubic-bezier)
- **Hovers**: Subtle scaling (1.1x) and rotation (5deg) on icons and primary buttons.

## Implementation Details

- All styles are defined in [style.css](../../app/static/css/style.css).
- No external CSS frameworks are used to maintain full control over the visual identity.
