# Chapter 5: Interface Design & Interaction

The CodaCite User Interface (UI) is the "Window into the Graph," designed to transform complex relational data into a high-density, intuitive workspace.

## 5.1 Design Philosophy: Functional Density

The CodaCite interface is designed for high-density information analysis, drawing inspiration from modern research tools like NotebookLM. It prioritizes:

* **Contextual Persistence**: Keeping the source documents visible while the user interacts with the AI.
* **Information Density**: Maximizing the visible data without overwhelming the user through a "Glassmorphism" design language.
* **Responsive Scaling**: The UI is optimized for a 1.5x zoom level, ensuring readability on high-resolution displays.

## 5.2 Key Interface Components

The application is divided into several high-functional zones:

1. **Notebook Sidebar**: Allows for rapid toggling between project-specific context scopes.
2. **Dynamic Knowledge Graph**: A real-time visualization of the entities and relationships extracted from the documents.
3. **Source-Grounded Chat**: A conversational interface where every response is anchored by clickable citation chips.

## 5.3 Design Tokens & Aesthetics

To maintain a premium, "textbook" feel, the UI utilizes a curated set of design tokens:

* **Typography**: Primary use of *Outfit* for headings and *Inter* for body text.
* **Colors**: A "Deep Sea" palette utilizing high-contrast HSL values for dark mode.
* **Glassmorphism**: Translucent surfaces with background blurs to maintain depth and hierarchy.

## 5.4 Interaction Heuristics

Functional density is achieved by mapping complex RAG operations to simple UI gestures:

* **Source Citations**: Clicking a citation chip instantly scrolls the document viewer to the exact `start_char` location.
* **Notebook Toggling**: Enabling/disabling notebooks triggers a reactive refresh of the underlying Graph Search scope.
* **Visual Evidence**: Entities in the Knowledge Graph are color-coded by type, matching their highlighting in the document viewer.
