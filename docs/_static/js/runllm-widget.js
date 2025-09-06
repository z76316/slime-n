// RunLLM Widget for Slime Documentation
(function() {
    // Create and append the RunLLM widget script
    const script = document.createElement('script');
    script.id = 'runllm-widget-script';
    script.type = 'module';
    script.crossOrigin = 'true';
    script.src = 'https://widget.runllm.com';
    
    // Configuration
    script.setAttribute('runllm-assistant-id', '1420');
    script.setAttribute('runllm-position', 'BOTTOM_RIGHT');
    script.setAttribute('runllm-name', 'Ask Slime');
    
    // Append to document
    document.head.appendChild(script);
})();