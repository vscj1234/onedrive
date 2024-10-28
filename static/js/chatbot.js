/* static/css/style.css */
body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                #chatbox { height: 300px; border: 1px solid #ddd; overflow-y: auto; padding: 10px; margin-bottom: 10px; }
                #userInput { width: 70%; padding: 5px; }
                button {
                    padding: 8px 15px;
                    background: #007bff;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }
                button:hover {
                    background: #0056b3;
                }
                #uploadSection {
                    margin: 20px 0;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background: #f5f5f5;
            }
            #fileInput {
                margin-right: 10px;
            }
            .upload-progress {
                margin-top: 10px;
                display: none;
            }
            .progress-bar {
                width: 100%;
                height: 20px;
                background-color: #f0f0f0;
                border-radius: 10px;
                overflow: hidden;
            }
            .progress-bar-fill {
                height: 100%;
                background-color: #4CAF50;
                width: 0%;
                transition: width 0.3s ease-in-out;
            }
            .message {
                margin: 5px 0;
                padding: 8px;
                border-radius: 4px;
            }
            .user-message {
                background: #e3f2fd;
                margin-left: 20%;
                margin-right: 5px;
            }
            .bot-message {
                background: #f5f5f5;
                margin-right: 20%;
                margin-left: 5px;
            }
            #relevantDocs { 
                margin-top: 20px;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            .doc-link { 
                color: #007bff;
                text-decoration: none;
                cursor: pointer;
            }
            .doc-link:hover {
                text-decoration: underline;
        }
            .upload-status {
                color: #666;
                font-style: italic;
                margin-top: 5px;
        }
