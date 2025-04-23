#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Componentes da interface do usuário para o sistema IndustriaCount
"""

import tkinter as tk
from tkinter import ttk

def create_ui(app):
    """
    Cria os componentes da interface do usuário para a aplicação
    
    Args:
        app: Instância da aplicação principal
    """
    main_frame = ttk.Frame(app.window, padding="10")
    main_frame.grid(row=0, column=0, sticky="nsew")
    main_frame.columnconfigure(0, weight=3)  # Video display area
    main_frame.columnconfigure(1, weight=1)  # Controls/Stats area
    main_frame.rowconfigure(0, weight=1)  # Make display row expand

    # === Left Column: Display ===
    display_frame = ttk.Frame(main_frame)
    display_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
    display_frame.rowconfigure(0, weight=1)
    display_frame.columnconfigure(0, weight=1)

    app.display_label = ttk.Label(display_frame, background='grey', cursor="crosshair")
    app.display_label.grid(row=0, column=0, pady=0, sticky="nsew")
    app.display_label.bind("<ButtonPress-1>", app.on_line_draw_start)
    app.display_label.bind("<B1-Motion>", app.on_line_draw_motion)
    app.display_label.bind("<ButtonRelease-1>", app.on_line_draw_end)

    # === Right Column: Controls & Info ===
    controls_info_frame = ttk.Frame(main_frame)
    controls_info_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
    controls_info_frame.columnconfigure(0, weight=1)
    # Configure rows for proportional space allocation
    controls_info_frame.rowconfigure(0, weight=0)  # Source Controls
    controls_info_frame.rowconfigure(1, weight=0)  # Model Config
    controls_info_frame.rowconfigure(2, weight=0)  # Line Config
    controls_info_frame.rowconfigure(3, weight=0)  # Status/Counts
    controls_info_frame.rowconfigure(4, weight=2)  # Statistics Display
    controls_info_frame.rowconfigure(5, weight=2)  # Plot
    controls_info_frame.rowconfigure(6, weight=1)  # Console

    # --- 1. Source Controls ---
    create_source_controls(app, controls_info_frame)

    # --- 2. Model Configuration ---
    create_model_controls(app, controls_info_frame)

    # --- 3. Line Configuration ---
    create_line_controls(app, controls_info_frame)

    # --- 4. Status & Counts ---
    create_status_display(app, controls_info_frame)

    # --- 5. Statistics Display ---
    create_stats_display(app, controls_info_frame)

    # --- 6. Plotting Frame ---
    create_plot_frame(app, controls_info_frame)

    # --- 7. Console ---
    create_console(app, controls_info_frame)

def create_source_controls(app, parent_frame):
    """Cria os controles de fonte de vídeo"""
    source_frame = ttk.LabelFrame(parent_frame, text="Fonte e Controle", padding="5")
    source_frame.grid(row=0, column=0, sticky="new", pady=(0, 5))
    source_frame.columnconfigure(0, weight=1)
    source_frame.columnconfigure(1, weight=1)
    
    app.btn_toggle_process = ttk.Button(source_frame, text="Iniciar Webcam", command=app.toggle_webcam)
    app.btn_toggle_process.grid(row=0, column=0, sticky="ew", padx=(0, 2), pady=2)
    
    app.btn_load_file = ttk.Button(source_frame, text="Carregar Arquivo", command=app.load_file)
    app.btn_load_file.grid(row=0, column=1, sticky="ew", padx=(2, 0), pady=2)
    
    app.btn_start_stop_monitor = ttk.Button(source_frame, text="Iniciar Monitoramento", 
                                           command=app.toggle_monitoring, state=tk.DISABLED)
    app.btn_start_stop_monitor.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)
    
    # Adicionar controles de otimização de desempenho
    perf_frame = ttk.LabelFrame(source_frame, text="Otimização de Desempenho")
    perf_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0))
    perf_frame.columnconfigure(1, weight=1)
    
    # Controle de pulo de frames
    ttk.Label(perf_frame, text="Pular frames:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    skip_frame_scale = ttk.Scale(perf_frame, from_=0, to=5, orient=tk.HORIZONTAL, 
                               variable=app.skip_frames)
    skip_frame_scale.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
    app.skip_frame_label = ttk.Label(perf_frame, text="0", width=3)
    app.skip_frame_label.grid(row=0, column=2, sticky="w", padx=5, pady=2)
    skip_frame_scale.configure(command=lambda v: app.skip_frame_label.configure(text=str(int(float(v)))))
    
    # Controle de escala de resolução
    ttk.Label(perf_frame, text="Resolução:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
    resolution_scale = ttk.Scale(perf_frame, from_=0.25, to=1.0, orient=tk.HORIZONTAL, 
                               variable=app.process_resolution)
    resolution_scale.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
    app.resolution_label = ttk.Label(perf_frame, text="100%", width=5)
    app.resolution_label.grid(row=1, column=2, sticky="w", padx=5, pady=2)
    resolution_scale.configure(command=lambda v: app.resolution_label.configure(
        text=f"{int(float(v)*100)}%"))

def create_model_controls(app, parent_frame):
    """Cria os controles de configuração do modelo"""
    model_frame = ttk.LabelFrame(parent_frame, text="Configuração do Modelo", padding="5")
    model_frame.grid(row=1, column=0, sticky="new", pady=5)
    model_frame.columnconfigure(1, weight=1)
    
    ttk.Button(model_frame, text="Carregar Modelo (.pt)", 
               command=app.load_model_dialog).grid(row=0, column=0, padx=5, pady=2, sticky="w")
    
    app.model_status_label = ttk.Label(model_frame, text="Modelo: Nenhum carregado", 
                                      anchor="w", relief=tk.SUNKEN, padding=(3, 1))
    app.model_status_label.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
    
    ttk.Label(model_frame, text="Confiança Mín.:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
    
    app.confidence_scale = ttk.Scale(model_frame, from_=0.1, to=0.95, orient=tk.HORIZONTAL, 
                                    variable=app.confidence_threshold, command=app._update_confidence_label)
    app.confidence_scale.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
    
    app.confidence_label = ttk.Label(model_frame, text=f"{app.confidence_threshold.get():.2f}", width=5)
    app.confidence_label.grid(row=1, column=2, sticky="w", padx=5, pady=2)
    
    # Adicionar controle para seleção de classes
    ttk.Label(model_frame, text="Classes:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
    
    # Frame para a lista de classes com scrollbar
    classes_frame = ttk.Frame(model_frame)
    classes_frame.grid(row=2, column=1, columnspan=2, sticky="ew", padx=5, pady=2)
    classes_frame.columnconfigure(0, weight=1)
    
    # Scrollable frame para checkboxes de classes
    app.classes_canvas = tk.Canvas(classes_frame, height=80)
    scrollbar = ttk.Scrollbar(classes_frame, orient="vertical", command=app.classes_canvas.yview)
    app.classes_scrollable_frame = ttk.Frame(app.classes_canvas)
    
    app.classes_scrollable_frame.bind(
        "<Configure>",
        lambda e: app.classes_canvas.configure(scrollregion=app.classes_canvas.bbox("all"))
    )
    
    app.classes_canvas.create_window((0, 0), window=app.classes_scrollable_frame, anchor="nw")
    app.classes_canvas.configure(yscrollcommand=scrollbar.set)
    
    app.classes_canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Botões para selecionar/deselecionar todas as classes
    select_buttons_frame = ttk.Frame(model_frame)
    select_buttons_frame.grid(row=3, column=1, columnspan=2, sticky="ew", padx=5, pady=2)
    
    ttk.Button(select_buttons_frame, text="Selecionar Todas", 
               command=app.select_all_classes).pack(side="left", padx=2)
    ttk.Button(select_buttons_frame, text="Limpar Seleção", 
               command=app.clear_class_selection).pack(side="left", padx=2)
    ttk.Button(select_buttons_frame, text="Aplicar", 
               command=app.apply_class_selection).pack(side="right", padx=2)

def create_line_controls(app, parent_frame):
    """Cria os controles de configuração da linha de contagem"""
    line_frame = ttk.LabelFrame(parent_frame, text="Linha de Contagem", padding="5")
    line_frame.grid(row=2, column=0, sticky="new", pady=5)
    line_frame.columnconfigure(1, weight=1)
    
    app.chk_draw_line = ttk.Checkbutton(line_frame, text="Habilitar Desenho da Linha", 
                                       variable=app.enable_line_drawing, command=app._on_toggle_draw_mode)
    app.chk_draw_line.grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=2)
    
    ttk.Label(line_frame, text="Direção Contagem:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
    
    app.combo_direction = ttk.Combobox(line_frame, textvariable=app.crossing_direction,
                                      values=["Esquerda -> Direita", "Direita -> Esquerda", 
                                              "Cima -> Baixo", "Baixo -> Cima"],
                                      state="readonly", width=18)
    app.combo_direction.grid(row=1, column=1, sticky="ew", padx=5, pady=2)

def create_status_display(app, parent_frame):
    """Cria a área de exibição de status e contagem"""
    status_frame = ttk.Frame(parent_frame)
    status_frame.grid(row=3, column=0, sticky="new", pady=5)
    status_frame.columnconfigure(0, weight=1)
    
    app.status_label = ttk.Label(status_frame, text="Status: Parado", anchor=tk.W)
    app.status_label.grid(row=0, column=0, sticky="ew", padx=5)
    
    app.count_label = ttk.Label(status_frame, text="Total Contado: 0", 
                               anchor=tk.W, font=('Arial', 10, 'bold'))
    app.count_label.grid(row=1, column=0, sticky="ew", padx=5)

def create_stats_display(app, parent_frame):
    """Cria a área de exibição de estatísticas"""
    stats_frame = ttk.LabelFrame(parent_frame, text="Estatísticas", padding="5")
    stats_frame.grid(row=4, column=0, sticky="nsew", pady=5)
    stats_frame.columnconfigure(0, weight=1)
    stats_frame.rowconfigure(0, weight=1)
    
    # Frame para estatísticas em tempo real
    realtime_frame = ttk.Frame(stats_frame)
    realtime_frame.grid(row=0, column=0, sticky="new", pady=(0, 5))
    realtime_frame.columnconfigure(0, weight=1)
    realtime_frame.columnconfigure(1, weight=1)
    
    # Taxa de contagem em tempo real
    ttk.Label(realtime_frame, text="Taxa de contagem:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    app.rate_label = ttk.Label(realtime_frame, text="0.0 obj/min", font=('Arial', 10, 'bold'))
    app.rate_label.grid(row=0, column=1, sticky="e", padx=5, pady=2)
    
    # Última classe contada
    ttk.Label(realtime_frame, text="Última classe:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
    app.last_class_label = ttk.Label(realtime_frame, text="Nenhuma")
    app.last_class_label.grid(row=1, column=1, sticky="e", padx=5, pady=2)
    
    # Última cor detectada
    ttk.Label(realtime_frame, text="Última cor:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
    app.last_color_label = ttk.Label(realtime_frame, text="Nenhuma")
    app.last_color_label.grid(row=2, column=1, sticky="e", padx=5, pady=2)
    
    # Separador
    ttk.Separator(stats_frame, orient="horizontal").grid(row=1, column=0, sticky="ew", pady=5)
    
    # Estatísticas detalhadas
    app.stats_output = tk.Text(stats_frame, height=6, state=tk.DISABLED, 
                              wrap=tk.WORD, relief=tk.SUNKEN, borderwidth=1, 
                              font=("Courier New", 8))
    app.stats_output.grid(row=2, column=0, sticky="nsew", pady=(0, 5))
    
    stats_scrollbar = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=app.stats_output.yview)
    stats_scrollbar.grid(row=2, column=1, sticky="ns", pady=(0, 5))
    app.stats_output['yscrollcommand'] = stats_scrollbar.set
    
    # Botões de controle
    control_frame = ttk.Frame(stats_frame)
    control_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
    control_frame.columnconfigure(0, weight=1)
    control_frame.columnconfigure(1, weight=1)
    control_frame.columnconfigure(2, weight=1)
    
    ttk.Button(control_frame, text="Resetar Estatísticas", 
               command=app.reset_statistics).grid(row=0, column=0, padx=2, pady=(0, 2), sticky="ew")
    
    ttk.Button(control_frame, text="Exportar JSON", 
               command=app.export_statistics).grid(row=0, column=1, padx=2, pady=(0, 2), sticky="ew")
               
    ttk.Button(control_frame, text="Exportar PDF", 
               command=app.export_pdf_report).grid(row=0, column=2, padx=2, pady=(0, 2), sticky="ew")

def create_plot_frame(app, parent_frame):
    """Cria a área de plotagem de gráficos"""
    plot_outer_frame = ttk.LabelFrame(parent_frame, text="Contagem Acumulada por Classe", padding="5")
    plot_outer_frame.grid(row=5, column=0, sticky="nsew", pady=5)
    plot_outer_frame.columnconfigure(0, weight=1)
    plot_outer_frame.rowconfigure(1, weight=1)
    
    # Dropdown to select class for plotting
    plot_controls_frame = ttk.Frame(plot_outer_frame)
    plot_controls_frame.grid(row=0, column=0, sticky="ew")
    
    ttk.Label(plot_controls_frame, text="Exibir Classe:").pack(side=tk.LEFT, padx=(0, 5))
    
    app.combo_plot_class = ttk.Combobox(plot_controls_frame, textvariable=app.selected_plot_class, 
                                       state="readonly", width=15)
    app.combo_plot_class.pack(side=tk.LEFT, fill=tk.X, expand=True)
    app.combo_plot_class.bind("<<ComboboxSelected>>", app.update_plot)  # Update plot on selection change
    
    # Plot Canvas Area
    app.plot_canvas_frame = ttk.Frame(plot_outer_frame)  # Frame to hold the canvas
    app.plot_canvas_frame.grid(row=1, column=0, sticky="nsew")
    app.plot_canvas_frame.columnconfigure(0, weight=1)
    app.plot_canvas_frame.rowconfigure(0, weight=1)

def create_console(app, parent_frame):
    """Cria a área de console para mensagens"""
    console_frame = ttk.LabelFrame(parent_frame, text="Mensagens", padding="5")
    console_frame.grid(row=6, column=0, sticky="nsew", pady=(5, 0))
    console_frame.columnconfigure(0, weight=1)
    console_frame.rowconfigure(0, weight=1)
    
    app.console_output = tk.Text(console_frame, height=4, state=tk.DISABLED, 
                                wrap=tk.WORD, relief=tk.SUNKEN, borderwidth=1)
    app.console_output.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
    
    console_scrollbar = ttk.Scrollbar(console_frame, orient=tk.VERTICAL, command=app.console_output.yview)
    console_scrollbar.grid(row=0, column=1, sticky="ns", pady=(0, 5))
    app.console_output['yscrollcommand'] = console_scrollbar.set
