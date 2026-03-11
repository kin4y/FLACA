def modern_style():
    """Returns the CSS for the modern DARK MODE UI."""
    return """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            /* Dark Mode Palette (Zinc) */
            --bg-body: #09090b;       /* Zinc 950 */
            --bg-panel: #18181b;      /* Zinc 900 */
            --bg-element: #27272a;    /* Zinc 800 */
            
            --text-main: #f4f4f5;     /* Zinc 100 */
            --text-sub: #a1a1aa;      /* Zinc 400 */
            
            --border: #3f3f46;        /* Zinc 700 */
            
            --primary: #0d9488;       /* Teal 600 */
            --primary-hover: #0f766e; /* Teal 700 */
            --accent: #ea580c;        /* Orange 600 */
            
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.3);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
            
            --radius: 0.5rem;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-body);
            color: var(--text-main);
            margin: 0;
            padding: 0;
            line-height: 1.6;
        }

        /* Header */
        header {
            background: var(--bg-panel);
            border-bottom: 1px solid var(--border);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: var(--shadow-sm);
        }
        header h1 {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-main);
            margin: 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        header h1 span { color: var(--primary); }
        
        /* Nav */
        nav {
            background: var(--bg-panel);
            padding: 0.5rem 2rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            gap: 0.5rem;
            justify-content: center;
        }
        nav a {
            color: var(--text-sub);
            text-decoration: none;
            padding: 0.5rem 1rem;
            border-radius: 0.375rem;
            font-weight: 500;
            font-size: 0.9rem;
            transition: all 0.2s;
        }
        nav a:hover {
            background-color: var(--bg-element);
            color: var(--text-main);
        }

        /* Layout */
        main {
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 1.5rem;
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 2rem;
            align-items: start;
        }
        main.full-width {
            grid-template-columns: 1fr;
            max-width: 1000px;
        }

        /* Panels */
        .panel {
            background: var(--bg-panel);
            border-radius: var(--radius);
            box-shadow: var(--shadow-md);
            padding: 1.5rem;
            border: 1px solid var(--border);
        }

        /* Sidebar Inputs */
        label {
            display: block;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
            color: var(--text-sub);
            margin-bottom: 0.4rem;
            margin-top: 1.2rem;
        }
        label:first-of-type { margin-top: 0; }

        input[type="text"], input[type="number"], select {
            width: 100%;
            padding: 0.6rem;
            background-color: var(--bg-body);
            border: 1px solid var(--border);
            color: var(--text-main);
            border-radius: 0.375rem;
            font-family: inherit;
            box-sizing: border-box;
        }
        input[type="text"]:focus, input[type="number"]:focus, select:focus {
            outline: 2px solid var(--primary);
            border-color: transparent;
        }
        input[type="file"] {
            font-size: 0.875rem;
            color: var(--text-sub);
            margin-top: 0.25rem;
        }

        /* Buttons */
        .btn, input[type="submit"], button {
            display: inline-flex;
            justify-content: center;
            align-items: center;
            padding: 0.6rem 1.2rem;
            border: none;
            border-radius: 0.375rem;
            background-color: var(--primary);
            color: white;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            transition: background-color 0.2s, transform 0.1s;
            text-decoration: none;
            width: 100%;
            box-sizing: border-box;
            margin-top: 1rem;
        }
        .btn:hover, input[type="submit"]:hover, button:hover {
            background-color: var(--primary-hover);
        }
        .btn-secondary {
            background-color: var(--bg-element);
            color: var(--text-main);
            border: 1px solid var(--border);
        }
        .btn-secondary:hover {
            background-color: var(--border);
        }
        .btn-danger {
            background-color: #7f1d1d; /* Dark red */
            color: #fecaca;
        }
        .btn-danger:hover { background-color: #991b1b; }

        /* Toolbars (Horizontal) */
        .toolbar {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }
        .toolbar .btn {
            width: auto;
            margin-top: 0;
            font-size: 0.8rem;
            padding: 0.4rem 0.8rem;
        }

        /* Swatches Grid */
        .swatch-container { 
            display: grid; 
            grid-template-columns: repeat(6, 1fr); 
            gap: 6px; 
            margin-top: 0.5rem; 
        }
        .swatch {
            width: 100%;
            aspect-ratio: 1;
            border-radius: 4px;
            cursor: pointer;
            border: 2px solid var(--border);
            transition: transform 0.1s;
        }
        .swatch:hover { transform: scale(1.1); z-index:2; border-color:white; }
        .swatch.active {
            border-color: white;
            box-shadow: 0 0 0 2px var(--primary);
            transform: scale(1.1);
            z-index:2;
        }

        /* Canvas */
        #canvas-container {
            width: 100%;
            height: 65vh;
            background-color: #000000;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
            position: relative;
            cursor: crosshair;
        }
        
        /* Info Boxes */
        .info-box {
            background-color: rgba(13, 148, 136, 0.1); /* Teal tint */
            border-left: 3px solid var(--primary);
            padding: 1rem;
            border-radius: 4px;
            font-size: 0.9rem;
            color: #ccfbf1; /* Teal 100 */
            margin: 1rem 0;
        }
        .info-box strong { color: white; }
        .info-box a { color: var(--primary); }
        
        /* Landing Page Grid */
        .landing-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-top: 3rem;
        }

        /* Gallery & Lightbox */
        /* Small gallery items for cluster inspections */
        .gallery-grid { \n            display: grid; \n            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); \n            gap: 10px; \n            margin-top: 15px; \n        }
        .gallery-item { \n            position: relative; \n            cursor: pointer; \n            border: 1px solid var(--border); \n            border-radius: 4px; \n            overflow: hidden; \n            transition: all 0.2s; \n            background: var(--bg-element); \n        }
        .gallery-item:hover { border-color: var(--primary); transform: translateY(-2px); box-shadow: var(--shadow-sm); }
        .gallery-item img { width: 100%; height: 100px; object-fit: cover; display: block; }
        .gallery-item span { \n            position: absolute; \n            bottom: 0; left: 0; right: 0; \n            background: rgba(0,0,0,0.7); \n            color: white; \n            font-size: 0.7rem; \n            padding: 2px 5px; \n            text-align: center;\n        }
        
        /* Large Primary Plot Items */
        .gallery-item-large {
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
            position: relative;
            cursor: pointer;
            transition: all 0.2s;
            background: #000; /* Plots are on black background */
        }
        .gallery-item-large:hover {
            border-color: var(--primary);
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }
        .gallery-item-large img {
            width: 100%;
            height: auto; 
            max-height: 400px; /* Max size for core plots */
            object-fit: contain;
            display: block;
        }
        .gallery-item-large span {
            position: absolute;
            top: 0; left: 0; 
            background: rgba(0,0,0,0.8);
            color: var(--text-main);
            font-size: 0.8rem;
            padding: 4px 8px;
            border-bottom-right-radius: 4px;
        }

        #lightbox { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.95); z-index: 9999; display: flex; justify-content: center; align-items: center; visibility: hidden; opacity: 0; transition: opacity 0.2s; }
        #lightbox.active { visibility: visible; opacity: 1; }
        #lightbox-canvas { max-width: 95%; max-height: 95%; cursor: grab; }
        #lightbox-canvas:active { cursor: grabbing; }
        #lightbox-close { position: absolute; top: 20px; right: 30px; color: white; font-size: 2rem; cursor: pointer; z-index: 10000; }
        #lightbox-info { position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); color: #aaa; font-size: 0.9rem; pointer-events: none; }
        
        /* Utilities */
        .hidden { display: none; }
        hr { border: 0; border-top: 1px solid var(--border); margin: 2rem 0; }
        footer {
            text-align: center;
            padding: 2rem;
            color: var(--text-sub);
            font-size: 0.8rem;
            margin-top: auto;
            border-top: 1px solid var(--border);
        }
    </style>
    <script>
        let lb_img = new Image();
        let lb_scale = 1, lb_originX = 0, lb_originY = 0;
        let lb_isPanning = false, lb_startX = 0, lb_startY = 0;
        
        function openLightbox(url) {
            const lb = document.getElementById('lightbox');
            const cvs = document.getElementById('lightbox-canvas');
            const ctx = cvs.getContext('2d');
            
            lb.classList.add('active');
            lb_img.src = url;
            lb_img.onload = () => {
                cvs.width = window.innerWidth;
                cvs.height = window.innerHeight;
                // Fit image
                lb_scale = Math.min((cvs.width-100)/lb_img.width, (cvs.height-100)/lb_img.height);
                lb_originX = (cvs.width - lb_img.width * lb_scale) / 2;
                lb_originY = (cvs.height - lb_img.height * lb_scale) / 2;
                drawLightbox();
            }
            
            // Event Listeners for Zoom/Pan
            cvs.onwheel = (e) => {
                e.preventDefault();
                const rect = cvs.getBoundingClientRect();
                const mx = e.clientX - rect.left;
                const my = e.clientY - rect.top;
                const worldX = (mx - lb_originX) / lb_scale;
                const worldY = (my - lb_originY) / lb_scale;
                
                const factor = e.deltaY < 0 ? 1.1 : 0.9;
                lb_scale *= factor;
                
                lb_originX = mx - worldX * lb_scale;
                lb_originY = my - worldY * lb_scale;
                drawLightbox();
            };
            
            cvs.onmousedown = (e) => {
                lb_isPanning = true;
                lb_startX = e.clientX - lb_originX;
                lb_startY = e.clientY - lb_originY;
                cvs.style.cursor = 'grabbing';
            };
            window.onmouseup = () => { lb_isPanning = false; document.getElementById('lightbox-canvas').style.cursor = 'grab'; };
            cvs.onmousemove = (e) => {
                if(!lb_isPanning) return;
                lb_originX = e.clientX - lb_startX;
                lb_originY = e.clientY - lb_startY;
                drawLightbox();
            };
            cvs.oncontextmenu = e => e.preventDefault();
            
            // ESC key to close
            document.onkeydown = (e) => {
                if (e.key === 'Escape' && lb.classList.contains('active')) {
                    closeLightbox();
                }
            };
        }
        
        function drawLightbox() {
            const cvs = document.getElementById('lightbox-canvas');
            const ctx = cvs.getContext('2d');
            ctx.clearRect(0,0,cvs.width, cvs.height);
            ctx.save();
            ctx.translate(lb_originX, lb_originY);
            ctx.scale(lb_scale, lb_scale);
            ctx.drawImage(lb_img, 0, 0);
            ctx.restore();
        }
        
        function closeLightbox() {
            document.getElementById('lightbox').classList.remove('active');
            window.onmouseup = null;
            document.onkeydown = null; 
        }
    </script>
    """

def page_layout(main, sidebar=None, show_nav=True):
    nav_html = """
    <nav>
        <a href="/">🏠 Home</a>
        <a href="/polygon_cutter">✂️ Polygon Cutter</a>
        <a href="/color_analysis">🎨 Color Analysis</a>
        <a href="/mesh_upload">☁️ 3D Mesh</a>
    </nav>
    """ if show_nav else ""
    
    content = f"""<main><div class="panel sidebar">{sidebar}</div><div class="panel analysis">{main}</div></main>""" if sidebar else f"""<main class="full-width"><div class="panel">{main}</div></main>"""
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Archaeological Analysis</title>
        {modern_style()}
    </head>
    <body>
        <header><h1><span>🏺</span> Archaeological Analysis</h1></header>
        {nav_html}
        {content}
        <div id="lightbox">
            <div id="lightbox-close" onclick="closeLightbox()">&times;</div>
            <canvas id="lightbox-canvas"></canvas>
            <div id="lightbox-info">Scroll to Zoom • Drag to Pan • ESC to Close</div>
        </div>
        <footer>Built for El Tajín, Veracruz</footer>
    </body>
    </html>
    """