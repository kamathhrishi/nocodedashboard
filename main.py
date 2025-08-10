import pandas as pd
import numpy as np
from openai import OpenAI
import json
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from jinja2 import Template
import re

class CSVDashboardGenerator:
    def __init__(self, api_key, csv_file_path):
        """
        Initialize the CSV Dashboard Generator
        
        Args:
            api_key (str): OpenAI API key
            csv_file_path (str): Path to the CSV file
        """
        self.api_key = api_key
        self.csv_file_path = csv_file_path
        self.df = None
        self.insights = None
        self.analysis_operations = None
        self.custom_prompt = None  # Store custom prompt from user
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
    
    def load_and_analyze_csv(self):
        """Load CSV file and perform basic analysis"""
        try:
            self.df = pd.read_csv(self.csv_file_path)
            print(f"✅ CSV loaded successfully. Shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"❌ Error loading CSV: {str(e)}")
            return False
    
    def get_data_summary(self):
        """Generate a comprehensive data summary"""
        summary = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "numeric_columns": list(self.df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(self.df.select_dtypes(include=['object']).columns),
            "sample_data": self.df.head(3).to_dict('records')
        }
        
        # Add basic statistics for numeric columns
        if summary["numeric_columns"]:
            summary["numeric_stats"] = self.df.describe().to_dict()
        
        return summary
    
    def generate_dashboard_recommendations(self, data_summary):
        """Use OpenAI to generate dashboard recommendations"""
        
        # Base prompt for dashboard recommendations
        base_prompt = f"""
        You are a data analyst expert who provides insightful dashboard recommendations based on dataset characteristics.
        
        Analyze this CSV dataset and provide dashboard recommendations:
        
        Dataset Summary:
        - Shape: {data_summary['shape'][0]} rows, {data_summary['shape'][1]} columns
        - Columns: {', '.join(data_summary['columns'])}
        - Numeric columns: {', '.join(data_summary['numeric_columns'])}
        - Categorical columns: {', '.join(data_summary['categorical_columns'])}
        - Missing values: {data_summary['missing_values']}
        - Sample data: {json.dumps(data_summary['sample_data'], indent=2)}
        """
        
        # Add custom prompt if provided by user
        if self.custom_prompt:
            custom_instruction = f"""
            
            USER'S SPECIFIC REQUIREMENTS:
            {self.custom_prompt}
            
            Please incorporate these specific requirements into your dashboard recommendations.
            """
            prompt = base_prompt + custom_instruction + """
            
            Please provide:
            1. What type of dashboard would be most suitable for this data
            2. Key metrics and KPIs to display
            3. Recommended visualizations (charts, graphs, tables)
            4. Insights that could be derived from this data
            5. Target audience for this dashboard
            
            Format your response as a detailed analysis in paragraph form.
            """
        else:
            prompt = base_prompt + """
            
            Please provide:
            1. What type of dashboard would be most suitable for this data
            2. Key metrics and KPIs to display
            3. Recommended visualizations (charts, graphs, tables)
            4. Insights that could be derived from this data
            5. Target audience for this dashboard
            
            Format your response as a detailed analysis in paragraph form.
            """
        
        try:
            response = self.client.responses.create(
                model="gpt-5",
                input=prompt,
                reasoning={
                    "effort": "minimal"
                },
                text={
                    "verbosity": "medium"
                }
            )
            
            self.insights = response.output_text
            print("✅ Dashboard recommendations generated")
            return self.insights
            
        except Exception as e:
            print(f"❌ Error generating recommendations: {str(e)}")
            return f"Error generating recommendations: {str(e)}"
    
    def generate_analysis_operations(self, data_summary):
        """Generate pandas operations for data analysis that create Plotly visualizations"""
        
        base_prompt = f"""
        You are a Python pandas and Plotly expert. Generate executable code that performs analysis AND creates Plotly visualizations.
        
        Dataset Info:
        - Columns: {', '.join(data_summary['columns'])}
        - Numeric columns: {', '.join(data_summary['numeric_columns'])}
        - Categorical columns: {', '.join(data_summary['categorical_columns'])}
        """
        
        if self.custom_prompt:
            custom_instruction = f"""
            
            USER'S SPECIFIC REQUIREMENTS:
            {self.custom_prompt}
            
            Please focus your analysis and visualizations on these specific requirements.
            """
            prompt = base_prompt + custom_instruction + """
            
            Generate Python code that:
            1. Performs comprehensive data analysis using pandas
            2. Creates interactive Plotly visualizations and stores them in a plots list
            3. Each plot should be stored as: plots.append(('plot_name', fig, 'explanation'))
            
            CRITICAL REQUIREMENTS:
            - Use 'df' as the DataFrame variable name
            - Import plotly.express as px and plotly.graph_objects as go
            - Create an empty list called 'plots = []' at the start
            - For each visualization, create the figure and add to plots list
            - DO NOT use matplotlib or plt.show() - ONLY use Plotly
            - Store plots as: plots.append(('descriptive_name', fig_object, 'what this chart shows'))
            - Include both analysis tables AND visualizations
            - Handle missing values and data type conversions
            - Use try-except blocks for operations that might fail
            
            Example format:
            # Import required libraries
            import plotly.express as px
            import plotly.graph_objects as go
            
            # Initialize plots list
            plots = []
            
            # Data analysis and visualization
            try:
                # Analysis code here
                analysis_result = df.groupby('column').mean()
                
                # Create Plotly visualization
                fig = px.bar(x=analysis_result.index, y=analysis_result.values, 
                            title='Analysis Title')
                plots.append(('analysis_chart', fig, 'This chart shows...'))
            except Exception as e:
                print(f"Error: {e}")
            """
        else:
            prompt = base_prompt + """
            
            Generate Python code that:
            1. Performs comprehensive data analysis using pandas
            2. Creates interactive Plotly visualizations and stores them in a plots list
            3. Each plot should be stored as: plots.append(('plot_name', fig, 'explanation'))
            
            CRITICAL REQUIREMENTS:
            - Use 'df' as the DataFrame variable name
            - Import plotly.express as px and plotly.graph_objects as go
            - Create an empty list called 'plots = []' at the start
            - For each visualization, create the figure and add to plots list
            - DO NOT use matplotlib or plt.show() - ONLY use Plotly
            - Store plots as: plots.append(('descriptive_name', fig_object, 'what this chart shows'))
            - Include both analysis tables AND visualizations
            - Handle missing values and data type conversions
            - Use try-except blocks for operations that might fail
            
            Example format:
            # Import required libraries
            import plotly.express as px
            import plotly.graph_objects as go
            
            # Initialize plots list
            plots = []
            
            # Data analysis and visualization
            try:
                # Analysis code here
                analysis_result = df.groupby('column').mean()
                
                # Create Plotly visualization
                fig = px.bar(x=analysis_result.index, y=analysis_result.values, 
                            title='Analysis Title')
                plots.append(('analysis_chart', fig, 'This chart shows...'))
            except Exception as e:
                print(f"Error: {e}")
            """
        
        try:
            response = self.client.responses.create(
                model="gpt-5",
                input=prompt,
                reasoning={
                    "effort": "minimal"
                },
                text={
                    "verbosity": "medium"
                }
            )
            
            self.analysis_operations = response.output_text
            print("✅ Analysis operations with Plotly visualizations generated")
            return self.analysis_operations
            
        except Exception as e:
            print(f"❌ Error generating analysis operations: {str(e)}")
            return f"# Error generating operations: {str(e)}"
    
    def execute_analysis_operations(self):
        """Execute pandas operations individually and capture results"""
        analysis_results = {}
        
        if not self.analysis_operations:
            return analysis_results
        
        print("🔄 Starting analysis execution with retry mechanism...")
        print("📋 Execution strategy: Bulk → Block-by-Block → Simplified → Basic")
        print("🔄 Retry mechanism: Up to 3 attempts per strategy with code correction")
        
        # Try different execution strategies with retry mechanism
        execution_strategies = [
            ("bulk", self._execute_bulk),
            ("block_by_block", self._execute_blocks_individually),
            ("simplified", self._execute_simplified_analysis),
            ("basic", self._create_basic_analysis_results)
        ]
        
        max_retries = 3
        current_attempt = 0
        
        for strategy_name, strategy_func in execution_strategies:
            current_attempt = 0
            while current_attempt < max_retries:
                try:
                    print(f"📊 Attempting {strategy_name} execution (attempt {current_attempt + 1}/{max_retries})...")
                    analysis_results = strategy_func()
                    
                    if analysis_results:
                        print(f"✅ {strategy_name.title()} execution successful")
                        return analysis_results
                    else:
                        print(f"⚠️ {strategy_name.title()} execution produced no results")
                        break  # Try next strategy instead of retrying
                        
                except Exception as e:
                    current_attempt += 1
                    print(f"❌ {strategy_name.title()} execution failed (attempt {current_attempt}/{max_retries}): {str(e)}")
                    
                    if current_attempt < max_retries:
                        print(f"🔄 Retrying with code correction...")
                        # Try to fix common issues in the code
                        if self._attempt_code_correction(str(e)):
                            continue  # Retry with corrected code
                        else:
                            # Code correction failed, try regeneration
                            print("🔄 Code correction failed, attempting to regenerate...")
                            if self._regenerate_analysis_operations():
                                current_attempt = 0  # Reset attempt counter
                                continue  # Try again with regenerated code
                    else:
                        print(f"💥 {strategy_name.title()} execution failed after {max_retries} attempts")
                        
                        # If this is the first strategy (bulk) and it failed, try to regenerate the code
                        if strategy_name == "bulk":
                            print("🔄 Attempting to regenerate analysis operations...")
                            if self._regenerate_analysis_operations():
                                # Reset attempt counter and try again with regenerated code
                                current_attempt = 0
                                continue
                        
                        continue  # Try next strategy
        
        print(f"✅ Analysis completed. Captured {len(analysis_results)} result blocks")
        return analysis_results
    
    def _attempt_code_correction(self, error):
        """Attempt to fix common code issues based on error messages"""
        try:
            error_str = str(error).lower()
            
            if "name 'df_clean' is not defined" in error_str:
                print("  🔧 Fixing: Replacing 'df_clean' with 'df' in analysis operations")
                self.analysis_operations = self.analysis_operations.replace('df_clean', 'df')
                
            elif "name" in error_str and "is not defined" in error_str:
                # Extract the undefined variable name
                import re
                match = re.search(r"name '([^']+)' is not defined", error_str)
                if match:
                    undefined_var = match.group(1)
                    print(f"  🔧 Fixing: Replacing undefined variable '{undefined_var}' with 'df'")
                    self.analysis_operations = self.analysis_operations.replace(undefined_var, 'df')
                    
            elif "keyerror" in error_str:
                print("  🔧 Fixing: Adding column existence checks to analysis operations")
                # Add basic column existence checks
                self.analysis_operations = self._add_column_checks(self.analysis_operations)
                
            elif "could not convert string to float" in error_str:
                print("  🔧 Fixing: Adding data type conversion handling")
                self.analysis_operations = self._add_type_conversion_handling(self.analysis_operations)
                
            # Test the corrected code
            if self._test_corrected_code():
                print("  ✅ Code correction applied and validated, retrying...")
                return True
            else:
                print("  ⚠️ Code correction failed validation, will try regeneration...")
                return False
            
        except Exception as correction_error:
            print(f"  ⚠️ Code correction failed: {str(correction_error)}")
            return False
    
    def _regenerate_analysis_operations(self):
        """Regenerate analysis operations with better error handling"""
        try:
            print("  🔄 Regenerating analysis operations with improved error handling...")
            
            # Create a more robust prompt that avoids common issues
            robust_prompt = f"""
            Generate Python pandas code for analyzing this dataset:
            
            Dataset Info:
            - Shape: {self.df.shape}
            - Columns: {list(self.df.columns)}
            - Numeric columns: {list(self.df.select_dtypes(include=[np.number]).columns)}
            - Categorical columns: {list(self.df.select_dtypes(include=['object']).columns)}
            
            CRITICAL REQUIREMENTS:
            - Use ONLY 'df' as the DataFrame variable name (NOT df_clean or any other names)
            - Always check if columns exist before using them
            - Handle data type conversions gracefully
            - Use try-except blocks for operations that might fail
            - Return ONLY valid Python code that can be executed
            - Focus on basic, reliable operations that won't fail
            
            Generate code for:
            1. Basic statistics for numeric columns
            2. Simple groupby operations (if applicable)
            3. Top values analysis
            4. Basic data overview
            """
            
            response = self.client.responses.create(
                model="gpt-5",
                input=robust_prompt,
                reasoning={
                    "effort": "minimal"
                },
                text={
                    "verbosity": "medium"
                }
            )
            
            self.analysis_operations = response.output_text
            print("  ✅ Analysis operations regenerated successfully")
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to regenerate analysis operations: {str(e)}")
            return False
    
    def _add_column_checks(self, code):
        """Add column existence checks to the code"""
        lines = code.split('\n')
        modified_lines = []
        
        for line in lines:
            modified_lines.append(line)
            
            # Add column checks after DataFrame operations
            if 'df[' in line and ']' in line and not line.strip().startswith('#'):
                # Extract column name from df[column] pattern
                import re
                match = re.search(r"df\[['\"]([^'\"]+)['\"]\]", line)
                if match:
                    column_name = match.group(1)
                    check_line = f"if '{column_name}' in df.columns:"
                    modified_lines.append(check_line)
                    # Indent the next line
                    if len(lines) > lines.index(line) + 1:
                        next_line = lines[lines.index(line) + 1]
                        if not next_line.strip().startswith('#'):
                            modified_lines.append(f"    {next_line}")
                            lines.pop(lines.index(line) + 1)  # Remove the next line since we've already added it
        
        return '\n'.join(modified_lines)
    
    def _add_type_conversion_handling(self, code):
        """Add data type conversion handling to the code"""
        lines = code.split('\n')
        modified_lines = []
        
        for line in lines:
            modified_lines.append(line)
            
            # Add type conversion for numeric operations
            if 'df[' in line and ']' in line and any(op in line for op in ['.mean()', '.sum()', '.describe()']):
                # Extract column name
                import re
                match = re.search(r"df\[['\"]([^'\"]+)['\"]\]", line)
                if match:
                    column_name = match.group(1)
                    # Add type conversion before the operation
                    conversion_line = f"df['{column_name}'] = pd.to_numeric(df['{column_name}'], errors='coerce')"
                    modified_lines.insert(len(modified_lines) - 1, conversion_line)
        
        return '\n'.join(modified_lines)
    
    def _execute_bulk(self):
        """Execute all operations as one block and capture both analysis and plots"""
        try:
            # Create execution environment with Plotly imports
            df = self.df.copy()
            local_vars = {
                'df': df, 
                'df_clean': df,
                'pd': pd, 
                'np': np,
                'px': px,
                'go': go,
                'plots': []  # Initialize plots list
            }
            
            # Execute the entire code block
            exec(self.analysis_operations, globals(), local_vars)
            
            # Capture generated plots
            if 'plots' in local_vars and local_vars['plots']:
                self.analysis_plots = local_vars['plots']
                print(f"✅ Captured {len(self.analysis_plots)} analysis plots")
            
            # Capture meaningful analysis results (excluding plots)
            meaningful_results = {}
            for var_name, var_value in local_vars.items():
                if var_name not in ['df', 'df_clean', 'pd', 'np', 'px', 'go', 'plots'] and not var_name.startswith('_'):
                    # Skip variables with generic names that are likely intermediate results
                    if var_name.lower() in ['summary', 'explanation', 'result', 'output', 'data', 'fig']:
                        continue
                    # Skip variables that contain problematic text patterns
                    var_str = str(var_value).lower()
                    if any(pattern in var_str for pattern in ['converted str to displayable format', 'str format for display purposes']):
                        continue
                    # Only include variables that are meaningful analysis results
                    if isinstance(var_value, (pd.DataFrame, pd.Series, int, float)):
                        meaningful_results[var_name] = var_value
                    elif isinstance(var_value, (list, tuple)) and len(var_value) > 0:
                        # Only include lists that seem like meaningful results
                        if not (isinstance(var_value[0], str) and any(col in str(var_value) for col in ['Revenue', 'Company', 'Sector'])):
                            meaningful_results[var_name] = var_value
                    elif isinstance(var_value, dict) and len(var_value) > 0:
                        # Only include dictionaries that seem like meaningful results
                        if not any(key in str(var_value) for key in ['dtype', 'Converted', 'str format']):
                            meaningful_results[var_name] = var_value
            
            # Capture results from meaningful variables only
            return self._capture_all_results(meaningful_results)
            
        except Exception as e:
            print(f"  ⚠️ Bulk execution failed: {str(e)}")
            raise e
    
    def _execute_blocks_individually(self):
        """Execute operations block by block"""
        try:
            # Create execution environment
            df = self.df.copy()
            local_vars = {
                'df': df, 
                'df_clean': df,  # Add df_clean as an alias to prevent undefined variable errors
                'pd': pd, 
                'np': np
            }
            
            # Parse operations into logical blocks
            operations_blocks = self._parse_operations_into_blocks(self.analysis_operations)
            
            print(f"  📊 Executing {len(operations_blocks)} analysis blocks individually...")
            
            analysis_results = {}
            for i, (block_name, code_block) in enumerate(operations_blocks):
                try:
                    print(f"    ⏳ Executing: {block_name}")
                    
                    # Execute the code block
                    exec(code_block, globals(), local_vars)
                    
                    # Only capture meaningful results from this block, not intermediate variables
                    meaningful_block_vars = {}
                    for var_name, var_value in local_vars.items():
                        if var_name not in ['df', 'df_clean', 'pd', 'np'] and not var_name.startswith('_'):
                            # Skip variables with generic names that are likely intermediate results
                            if var_name.lower() in ['summary', 'explanation', 'result', 'output', 'data']:
                                continue
                            # Skip variables that contain problematic text patterns
                            var_str = str(var_value).lower()
                            if any(pattern in var_str for pattern in ['converted str to displayable format', 'str format for display purposes']):
                                continue
                            # Only include variables that are meaningful analysis results
                            if isinstance(var_value, (pd.DataFrame, pd.Series, int, float)):
                                meaningful_block_vars[var_name] = var_value
                            elif isinstance(var_value, (list, tuple)) and len(var_value) > 0:
                                # Only include lists that seem like meaningful results
                                if not (isinstance(var_value[0], str) and any(col in str(var_value) for col in ['Revenue', 'Sector'])):
                                    meaningful_block_vars[var_name] = var_value
                            elif isinstance(var_value, dict) and len(var_value) > 0:
                                # Only include dictionaries that seem like meaningful results
                                if not any(key in str(var_value) for key in ['dtype', 'Converted', 'str format']):
                                    meaningful_block_vars[var_name] = var_value
                    
                    # Capture results from meaningful variables only
                    block_results = self._capture_block_results(meaningful_block_vars, block_name, code_block)
                    
                    if block_results:
                        analysis_results[f"block_{i}_{block_name}"] = block_results
                        print(f"    ✅ {block_name} completed")
                    
                except Exception as e:
                    print(f"    ⚠️ Error in {block_name}: {str(e)}")
                    # Provide helpful error information
                    self._provide_error_guidance(e)
                    continue
            
            return analysis_results
            
        except Exception as e:
            print(f"  ❌ Block execution failed: {str(e)}")
            raise e
    
    def _execute_simplified_analysis(self):
        """Execute a simplified version of the analysis"""
        try:
            # Create execution environment
            df = self.df.copy()
            local_vars = {
                'df': df, 
                'pd': pd, 
                'np': np
            }
            
            # Create meaningful analysis results directly without intermediate variables
            meaningful_results = {}
            
            # Basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                try:
                    df_stats = df[numeric_cols].describe()
                    meaningful_results['df_stats'] = df_stats
                except Exception as e:
                    print(f"    ⚠️ Basic statistics failed: {str(e)}")
            
            # Basic groupby if categorical columns exist
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols and numeric_cols:
                for cat_col in categorical_cols[:3]:  # Limit to first 3 categorical columns
                    for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                        try:
                            # Ensure numeric column is actually numeric
                            temp_df = df.copy()
                            temp_df[num_col] = pd.to_numeric(temp_df[num_col], errors='coerce')
                            temp_df = temp_df.dropna(subset=[num_col])
                            
                            if len(temp_df) > 0:
                                group_result = temp_df.groupby(cat_col)[num_col].mean()
                                if len(group_result) > 0:
                                    meaningful_results[f'group_{cat_col}_{num_col}'] = group_result
                        except Exception as e:
                            print(f"    ⚠️ Groupby {cat_col} by {num_col} failed: {str(e)}")
                            continue
            
            # Top values for key columns
            if numeric_cols:
                for col in numeric_cols[:3]:
                    try:
                        # Use available columns for display
                        display_cols = [col]
                        if 'Company' in df.columns:
                            display_cols.insert(0, 'Company')
                        elif categorical_cols:
                            display_cols.insert(0, categorical_cols[0])
                        
                        # Ensure numeric column is actually numeric
                        temp_df = df.copy()
                        temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
                        temp_df = temp_df.dropna(subset=[col])
                        
                        if len(temp_df) > 0:
                            top_values = temp_df.nlargest(10, col)[display_cols]
                            meaningful_results[f'top_{col}'] = top_values
                    except Exception as e:
                        print(f"    ⚠️ Top values for {col} failed: {str(e)}")
                        continue
            
            # Capture results from meaningful variables only
            return self._capture_all_results(meaningful_results)
            
        except Exception as e:
            print(f"  ❌ Simplified analysis failed: {str(e)}")
            raise e
    
    def _provide_error_guidance(self, error):
        """Provide helpful guidance based on error type"""
        error_str = str(error).lower()
        
        if "could not convert string to float" in error_str:
            print(f"       💡 Tip: This error usually means the data contains text in numeric columns")
        elif "invalid syntax" in error_str:
            print(f"       💡 Tip: The AI-generated code had syntax errors")
        elif "name" in error_str and "is not defined" in error_str:
            print(f"       💡 Tip: Variable dependency issue - some variables may not be available yet")
        elif "keyerror" in error_str:
            print(f"       💡 Tip: Column name not found - check if the column exists in your dataset")
        elif "typeerror" in error_str:
            print(f"       💡 Tip: Data type mismatch - the operation may not be suitable for this column type")
        else:
            print(f"       💡 Tip: Check the data types and column names in your dataset")
    
    def _parse_operations_into_blocks(self, operations_code):
        """Parse operations code into logical blocks with descriptive names"""
        blocks = []
        lines = operations_code.split('\n')
        current_block = []
        current_name = "Data Overview"
        
        for line in lines:
            line = line.strip()
            if line.startswith('#') and 'Basic statistics' in line:
                if current_block:
                    blocks.append((current_name, '\n'.join(current_block)))
                current_block = []
                current_name = "Statistical Summary"
            elif line.startswith('#') and 'Groupby' in line:
                if current_block:
                    blocks.append((current_name, '\n'.join(current_block)))
                current_block = []
                current_name = "Group Analysis"
            elif line.startswith('#') and 'filtering' in line:
                if current_block:
                    blocks.append((current_name, '\n'.join(current_block)))
                current_block = []
                current_name = "Data Filtering & Sorting"
            elif line.startswith('#') and 'Correlation' in line:
                if current_block:
                    blocks.append((current_name, '\n'.join(current_block)))
                current_block = []
                current_name = "Correlation Analysis"
            elif line.startswith('#') and ('Top' in line or 'bottom' in line or 'performers' in line):
                if current_block:
                    blocks.append((current_name, '\n'.join(current_block)))
                current_block = []
                current_name = "Top & Bottom Performers"
            elif line and not line.startswith('#'):
                current_block.append(line)
        
        # Add the last block
        if current_block:
            blocks.append((current_name, '\n'.join(current_block)))
        
        # Filter out empty blocks and validate syntax
        valid_blocks = []
        for name, code in blocks:
            if code.strip() and self._validate_python_syntax(code):
                valid_blocks.append((name, code))
        
        return valid_blocks
    
    def _validate_python_syntax(self, code):
        """Basic Python syntax validation"""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False
    
    def _capture_block_results(self, local_vars, block_name, code_block):
        """Capture and format results from executed code block with descriptive names"""
        results = {}
        
        # Look for DataFrames and Series that were created
        for var_name, var_value in local_vars.items():
            if var_name not in ['df', 'pd', 'np'] and not var_name.startswith('_'):
                # Skip variables with generic names that are likely intermediate results
                if var_name.lower() in ['summary', 'explanation', 'result', 'output', 'data']:
                    continue
                # Skip variables that contain problematic text patterns
                var_str = str(var_value).lower()
                if any(pattern in var_str for pattern in ['converted str to displayable format', 'str format for display purposes']):
                    continue
                try:
                    if isinstance(var_value, pd.DataFrame):
                        # Convert DataFrame to HTML table
                        if len(var_value) > 0:
                            # Limit size for display
                            display_df = var_value.head(10) if len(var_value) > 10 else var_value
                            
                            # Create descriptive name and summary based on block context
                            descriptive_name = self._create_descriptive_name(var_name, block_name, var_value)
                            descriptive_summary = self._create_descriptive_summary(var_name, block_name, var_value)
                            
                            results[descriptive_name] = {
                                'type': 'dataframe',
                                'shape': var_value.shape,
                                'html': display_df.round(2).to_html(classes='table table-striped table-hover table-sm'),
                                'summary': descriptive_summary,
                                'explanation': self._get_dataframe_explanation(block_name, var_value)
                            }
                    
                    elif isinstance(var_value, pd.Series):
                        # Convert Series to readable format
                        if len(var_value) > 0:
                            display_series = var_value.head(10) if len(var_value) > 10 else var_value
                            
                            # Create descriptive name and summary
                            descriptive_name = self._create_descriptive_name(var_name, block_name, var_value)
                            descriptive_summary = self._create_descriptive_summary(var_name, block_name, var_value)
                            
                            results[descriptive_name] = {
                                'type': 'series',
                                'length': len(var_value),
                                'html': display_series.round(2).to_frame().to_html(classes='table table-striped table-hover table-sm'),
                                'summary': descriptive_summary,
                                'explanation': self._get_series_explanation(block_name, var_value)
                            }
                    
                    elif isinstance(var_value, (int, float, str)):
                        # Simple scalar values
                        descriptive_name = self._create_descriptive_name(var_name, block_name, var_value)
                        descriptive_summary = self._create_descriptive_summary(var_name, block_name, var_value)
                        
                        results[descriptive_name] = {
                            'type': 'scalar',
                            'value': var_value,
                            'summary': descriptive_summary,
                            'explanation': self._get_scalar_explanation(block_name, var_value)
                        }
                
                except Exception as e:
                    continue
        
        return results
    
    def _create_descriptive_name(self, var_name, block_name, var_value):
        """Create a descriptive name for variables based on context"""
        # Map common variable names to descriptive names
        name_mapping = {
            'df': 'Complete Dataset',
            'df_filtered': 'Filtered Results',
            'df_sorted': 'Ranked Data',
            'df_grouped': 'Grouped Analysis',
            'df_corr': 'Relationship Matrix',
            'df_stats': 'Key Statistics',
            'df_top': 'Best Performers',
            'df_bottom': 'Areas for Improvement',
            'df_clean': 'Cleaned Data',
            'df_agg': 'Summary by Category'
        }
        
        # Use mapping if available, otherwise create descriptive name
        if var_name in name_mapping:
            return name_mapping[var_name]
        
        # Create descriptive name based on block context
        if 'filter' in var_name.lower():
            return 'Filtered Results'
        elif 'sort' in var_name.lower():
            return 'Ranked Data'
        elif 'group' in var_name.lower():
            return 'Grouped Analysis'
        elif 'corr' in var_name.lower():
            return 'Relationship Analysis'
        elif 'stats' in var_name.lower():
            return 'Key Statistics'
        elif 'top' in var_name.lower():
            return 'Best Performers'
        elif 'bottom' in var_name.lower():
            return 'Areas for Improvement'
        elif 'mean' in var_name.lower():
            return 'Average Values'
        elif 'sum' in var_name.lower():
            return 'Total Values'
        elif 'count' in var_name.lower():
            return 'Count Analysis'
        else:
            return var_name.replace('_', ' ').title()
    
    def _create_descriptive_summary(self, var_name, block_name, var_value):
        """Create a descriptive summary for variables"""
        if isinstance(var_value, pd.DataFrame):
            return f"Dataset with {var_value.shape[0]:,} records and {var_value.shape[1]} columns"
        elif isinstance(var_value, pd.Series):
            return f"Data series with {len(var_value):,} values"
        elif isinstance(var_value, (int, float)):
            return f"Calculated value: {var_value:,.2f}" if isinstance(var_value, float) else f"Calculated value: {var_value:,}"
        else:
            return f"Result: {var_value}"
    
    def _get_dataframe_explanation(self, block_name, df):
        """Generate explanation for DataFrame results"""
        explanations = {
            'Statistical Summary': f"This table shows the key numbers that summarize your data - like averages, typical ranges, and how spread out your values are. It gives you a quick overview of what's normal in your dataset.",
            'Data Filtering & Sorting': f"This shows {df.shape[0]:,} records that match specific criteria you're interested in. It's like filtering your data to focus on the most relevant information for your analysis.",
            'Group Analysis': f"This groups your data by categories and shows totals or averages for each group. It reveals patterns like which categories are performing best or how different groups compare to each other.",
            'Correlation Analysis': f"This shows how different columns in your data relate to each other. Values closer to 1 or -1 mean strong relationships, while values closer to 0 mean weak or no relationship.",
            'Top & Bottom Performers': f"This highlights your best and worst performing items based on key metrics. It helps you see what's working well and what might need attention or improvement."
        }
        
        return explanations.get(block_name, f"This analysis shows {df.shape[0]:,} records with {df.shape[1]} different pieces of information, giving you insights into patterns and trends in your data.")
    
    def _get_series_explanation(self, block_name, series):
        """Generate explanation for Series results"""
        explanations = {
            'Statistical Summary': "This shows key numbers for a specific column in your data, helping you understand what's typical and how much variation exists in that particular field.",
            'Data Filtering & Sorting': "This shows values from a specific column that meet certain criteria, making it easier to focus on the data points that matter most for your analysis.",
            'Group Analysis': "This shows totals or averages for different categories in a specific column, revealing which groups are performing better or worse.",
            'Correlation Analysis': "This shows how strongly one column relates to others in your data, helping you identify which factors might influence each other.",
            'Top & Bottom Performers': "This ranks items by their performance in a specific column, helping you quickly see who's leading and who might need support."
        }
        
        return explanations.get(block_name, f"This analysis shows {len(series):,} data points from a specific column, giving you insights into patterns and trends in that particular field.")
    
    def _get_scalar_explanation(self, block_name, value):
        """Generate explanation for scalar results"""
        explanations = {
            'Statistical Summary': f"This number ({value}) is a key summary statistic from your data - like an average, total, or typical value that helps you understand what's normal in your dataset.",
            'Data Filtering & Sorting': f"This result ({value}) shows what you get when you apply specific filters or sorting to your data, helping you focus on the most relevant information.",
            'Group Analysis': f"This number ({value}) represents the total or average for a specific group in your data, showing how that category performs compared to others.",
            'Correlation Analysis': f"This correlation value ({value}) shows how strongly two things in your data relate to each other - closer to 1 or -1 means stronger relationships.",
            'Top & Bottom Performers': f"This metric ({value}) is a performance score that helps rank items in your data, showing which ones are doing best or worst."
        }
        
        return explanations.get(block_name, f"This calculated result ({value}) is an important number that reveals key insights about your data and helps you understand patterns and trends.")
    
    def generate_analysis_insights(self, analysis_results):
        """Use AI to generate insights from analysis results"""
        if not analysis_results:
            return "No analysis results available."
        
        # Prepare summary of results for AI
        results_summary = {}
        for block_name, block_results in analysis_results.items():
            # Ensure block_results is a dictionary
            if not isinstance(block_results, dict):
                print(f"⚠️ Warning: block_results for {block_name} is not a dictionary: {type(block_results)}")
                continue
                
            block_summary = {}
            for var_name, var_data in block_results.items():
                # Ensure var_data is a dictionary
                if not isinstance(var_data, dict):
                    print(f"⚠️ Warning: var_data for {var_name} in {block_name} is not a dictionary: {type(var_data)}")
                    continue
                    
                # Check if the required keys exist
                if 'type' not in var_data:
                    print(f"⚠️ Warning: var_data for {var_name} in {block_name} missing 'type' key")
                    continue
                    
                if var_data['type'] == 'dataframe':
                    if 'shape' in var_data and 'summary' in var_data:
                        block_summary[var_name] = f"DataFrame ({var_data['shape'][0]}x{var_data['shape'][1]}): {var_data['summary']}"
                    else:
                        block_summary[var_name] = f"DataFrame: {var_data.get('summary', 'No summary available')}"
                elif var_data['type'] == 'series':
                    if 'length' in var_data and 'summary' in var_data:
                        block_summary[var_name] = f"Series ({var_data['length']} values): {var_data['summary']}"
                    else:
                        block_summary[var_name] = f"Series: {var_data.get('summary', 'No summary available')}"
                elif var_data['type'] == 'scalar':
                    if 'value' in var_data:
                        block_summary[var_name] = f"Value: {var_data['value']}"
                    else:
                        block_summary[var_name] = f"Scalar: {var_data.get('summary', 'No summary available')}"
                else:
                    # Handle unknown types gracefully
                    block_summary[var_name] = f"Unknown type ({var_data['type']}): {var_data.get('summary', 'No summary available')}"
            
            if block_summary:  # Only add if we have valid data
                results_summary[block_name] = block_summary
        
        # Base prompt for analysis insights
        base_prompt = f"""
        You are a business analyst expert. Based on the following pandas analysis results, generate 3-5 key business insights that would be valuable to executives and analysts.
        
        Analysis Results Summary:
        {json.dumps(results_summary, indent=2)}
        """
        
        # Add custom prompt if provided by user
        if self.custom_prompt:
            custom_instruction = f"""
            
            USER'S SPECIFIC REQUIREMENTS:
            {self.custom_prompt}
            
            Please focus your insights on these specific requirements while maintaining comprehensive business value.
            """
            prompt = base_prompt + custom_instruction + """
            
            Please provide:
            1. Top 3-5 most important insights from this analysis
            2. What these findings mean for business strategy
            3. Any patterns or trends that stand out
            4. Actionable recommendations based on the data
            
            Format your response as clear, concise bullet points that are easy to understand for business users.
            Focus on insights that would be relevant for financial performance, competitive positioning, and strategic decision making.
            """
        else:
            prompt = base_prompt + """
            
            Please provide:
            1. Top 3-5 most important insights from this analysis
            2. What these findings mean for business strategy
            3. Any patterns or trends that stand out
            4. Actionable recommendations based on the data
            
            Format your response as clear, concise bullet points that are easy to understand for business users.
            Focus on insights that would be relevant for financial performance, competitive positioning, and strategic decision making.
            """
        
        try:
            response = self.client.responses.create(
                model="gpt-5",
                input=prompt,
                reasoning={
                    "effort": "minimal"
                },
                text={
                    "verbosity": "medium"
                }
            )
            
            return response.output_text
            
        except Exception as e:
            print(f"❌ Error generating analysis insights: {str(e)}")
            return "Unable to generate insights from analysis results."
    
    def generate_smart_visualizations(self):
        """Use AI to generate contextually appropriate visualizations"""
        data_summary = self.get_data_summary()
        
        # Create a JSON-safe version of the data summary
        safe_summary = {
            "columns": data_summary["columns"],
            "numeric_columns": data_summary["numeric_columns"],
            "categorical_columns": data_summary["categorical_columns"],
            "shape": data_summary["shape"],
            "dtypes": {k: str(v) for k, v in data_summary["dtypes"].items()},
            "sample_data": []
        }
        
        # Convert sample data to JSON-safe format
        for record in data_summary["sample_data"]:
            safe_record = {}
            for k, v in record.items():
                if pd.isna(v):
                    safe_record[k] = None
                elif isinstance(v, (int, float, str, bool)):
                    safe_record[k] = v
                else:
                    safe_record[k] = str(v)
            safe_summary["sample_data"].append(safe_record)
        
        # Base prompt for visualization recommendations
        base_prompt = f"""
        You are a data visualization expert. Based on this dataset, recommend the most meaningful and insightful visualizations.
        
        Dataset Analysis:
        - Columns: {', '.join(safe_summary['columns'])}
        - Numeric columns: {', '.join(safe_summary['numeric_columns'])}
        - Categorical columns: {', '.join(safe_summary['categorical_columns'])}
        - Data types: {json.dumps(safe_summary['dtypes'], indent=2)}
        - Sample data: {json.dumps(safe_summary['sample_data'], indent=2)}
        """
        
        # Add custom prompt if provided by user
        if self.custom_prompt:
            custom_instruction = f"""
            
            USER'S SPECIFIC REQUIREMENTS:
            {self.custom_prompt}
            
            Please focus your visualization recommendations on these specific requirements while maintaining comprehensive coverage.
            """
            prompt = base_prompt + custom_instruction + """
            
            Suggest 4-6 specific visualizations that would provide maximum business value and insights. For each visualization, specify:
            1. Chart type (bar, line, scatter, pie, heatmap, box, etc.)
            2. X and Y axes (column names)
            3. Purpose/insight it reveals
            4. Any grouping or filtering needed
            
            Focus on charts that would be most relevant to business users analyzing this data.
            Return your response as a JSON array with this structure:
            [
                {
                    "chart_type": "bar",
                    "x_axis": "column_name",
                    "y_axis": "column_name",
                    "title": "Chart Title",
                    "purpose": "What insight this reveals",
                    "grouping": "optional_grouping_column"
                }
            ]
            """
        else:
            prompt = base_prompt + """
            
            Suggest 4-6 specific visualizations that would provide maximum business value and insights. For each visualization, specify:
            1. Chart type (bar, line, scatter, pie, heatmap, box, etc.)
            2. X and Y axes (column names)
            3. Purpose/insight it reveals
            4. Any grouping or filtering needed
            
            Focus on charts that would be most relevant to business users analyzing this data.
            Return your response as a JSON array with this structure:
            [
                {
                    "chart_type": "bar",
                    "x_axis": "column_name",
                    "y_axis": "column_name",
                    "title": "Chart Title",
                    "purpose": "What insight this reveals",
                    "grouping": "optional_grouping_column"
                }
            ]
            """
        
        try:
            response = self.client.responses.create(
                model="gpt-5",
                input=prompt,
                reasoning={
                    "effort": "minimal"
                },
                text={
                    "verbosity": "medium"
                }
            )
            
            # Parse the JSON response
            try:
                recommendations = json.loads(response.output_text)
                return recommendations
            except json.JSONDecodeError as json_error:
                print(f"❌ Error parsing JSON response: {str(json_error)}")
                print(f"📝 Raw response: {response.output_text[:200]}...")
                return []
            
        except Exception as e:
            print(f"❌ Error generating smart visualizations: {str(e)}")
            return []
    
    def create_visualizations(self):
        """Create plotly visualizations from both AI analysis and standard recommendations"""
        figs = []
        
        try:
            # First, add any plots generated during analysis execution
            if hasattr(self, 'analysis_plots') and self.analysis_plots:
                print(f"📊 Adding {len(self.analysis_plots)} plots from analysis...")
                for plot_name, fig, explanation in self.analysis_plots:
                    try:
                        # Apply professional styling to analysis plots
                        fig.update_layout(
                            font=dict(family="Arial, sans-serif", size=12),
                            title_font=dict(size=16, family="Arial, sans-serif"),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            showlegend=True,
                            margin=dict(l=50, r=50, t=80, b=50),
                            height=400
                        )
                        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
                        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
                        
                        figs.append((plot_name, fig.to_html(include_plotlyjs='cdn'), explanation))
                    except Exception as e:
                        print(f"⚠️ Could not process analysis plot {plot_name}: {str(e)}")
                        continue
            
            # Then, get AI-powered visualization recommendations for additional charts
            viz_recommendations = self.generate_smart_visualizations()
            
            # If AI recommendations fail, fall back to basic visualizations
            if not viz_recommendations:
                viz_recommendations = self._get_fallback_visualizations()
            
            # Validate visualization recommendations
            valid_recommendations = []
            for viz in viz_recommendations:
                if isinstance(viz, dict) and 'chart_type' in viz and 'x_axis' in viz:
                    valid_recommendations.append(viz)
                else:
                    print(f"⚠️ Skipping invalid visualization recommendation: {viz}")
            
            # Create additional charts from recommendations (avoid duplicates)
            existing_chart_types = set()
            if self.analysis_plots:
                # Try to identify chart types from existing plots to avoid duplication
                for plot_name, _, _ in self.analysis_plots:
                    if 'bar' in plot_name.lower():
                        existing_chart_types.add('bar')
                    elif 'scatter' in plot_name.lower():
                        existing_chart_types.add('scatter')
                    elif 'line' in plot_name.lower():
                        existing_chart_types.add('line')
            
            for i, viz in enumerate(valid_recommendations):
                # Skip if we already have similar chart types from analysis
                if viz.get('chart_type', '').lower() in existing_chart_types:
                    continue
                    
                try:
                    fig = self._create_chart_from_recommendation(viz)
                    if fig:
                        explanation = self._create_chart_explanation(viz, fig)
                        figs.append((f"additional_chart_{i}", fig.to_html(include_plotlyjs='cdn'), explanation))
                except Exception as e:
                    print(f"⚠️ Could not create visualization {i}: {str(e)}")
                    continue
            
            # Always include a correlation heatmap if we have multiple numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1 and not any('correlation' in name.lower() for name, _, _ in figs):
                try:
                    # Filter out non-numeric data that might cause correlation issues
                    clean_numeric_cols = []
                    for col in numeric_cols:
                        try:
                            pd.to_numeric(self.df[col], errors='coerce')
                            if not self.df[col].isna().all():
                                clean_numeric_cols.append(col)
                        except:
                            continue
                    
                    if len(clean_numeric_cols) > 1:
                        corr_data = self.df[clean_numeric_cols].apply(pd.to_numeric, errors='coerce').dropna()
                        if len(corr_data) > 0:
                            corr_matrix = corr_data.corr()
                            fig_corr = px.imshow(corr_matrix, 
                                            text_auto=True, 
                                            aspect="auto",
                                            title="Correlation Analysis - Variable Relationships",
                                            color_continuous_scale="RdBu_r")
                            fig_corr.update_layout(
                                font=dict(family="Arial, sans-serif", size=12),
                                title_font=dict(size=16, family="Arial, sans-serif"),
                                showlegend=True,
                                height=400
                            )
                            explanation = f"This correlation heatmap shows how different numeric variables in your {os.path.basename(self.csv_file_path).replace('.csv', '')} dataset relate to each other. Darker red colors indicate strong positive relationships (when one variable increases, the other tends to increase too), while darker blue colors show strong negative relationships (when one increases, the other decreases). Lighter colors indicate weaker relationships. This helps you identify which variables work together and which might be independent."
                            figs.append(("correlation_heatmap", fig_corr.to_html(include_plotlyjs='cdn'), explanation))
                except Exception as e:
                    print(f"⚠️ Could not create correlation heatmap: {str(e)}")
        
        except Exception as e:
            print(f"⚠️ Error in visualization creation: {str(e)}")
            # Fallback to basic charts
            figs = self._create_basic_visualizations()
        
        return figs
    
    def _create_chart_explanation(self, viz_rec, fig):
        """Create a specific explanation for a chart based on its data and purpose"""
        chart_type = viz_rec.get('chart_type', '').lower()
        x_axis = viz_rec.get('x_axis')
        y_axis = viz_rec.get('y_axis')
        title = viz_rec.get('title', 'Data Analysis')
        grouping = viz_rec.get('grouping')
        
        # Get dataset name for context
        dataset_name = os.path.basename(self.csv_file_path).replace('.csv', '')
        
        if chart_type == 'bar':
            if y_axis and pd.api.types.is_numeric_dtype(self.df[y_axis]):
                explanation = f"This bar chart shows the total {y_axis} for each {x_axis} category in your {dataset_name} dataset. The bars are arranged from highest to lowest, making it easy to see which {x_axis} categories have the highest {y_axis} values. This visualization helps you identify top performers and spot opportunities for improvement."
            else:
                explanation = f"This bar chart displays the frequency count of each {x_axis} category in your {dataset_name} dataset. It shows how many times each category appears, helping you understand the distribution and identify the most common {x_axis} values. This is useful for understanding your data's composition."
        
        elif chart_type == 'line':
            explanation = f"This line chart tracks how {y_axis} changes over {x_axis} in your {dataset_name} dataset. It reveals trends, patterns, and any seasonal or cyclical behavior. The line connects data points chronologically, making it easy to spot upward or downward trends over time."
        
        elif chart_type == 'scatter':
            if grouping and grouping in self.df.columns:
                explanation = f"This scatter plot shows the relationship between {x_axis} and {y_axis} in your {dataset_name} dataset, with points colored by {grouping}. It helps you see if there's a correlation between these two variables and whether the relationship differs across {grouping} categories. Clusters or patterns in the data can reveal important insights."
            else:
                explanation = f"This scatter plot explores the relationship between {x_axis} and {y_axis} in your {dataset_name} dataset. Each point represents one data record. If points form a clear pattern (like a line going up or down), it suggests a relationship between these variables. Random scatter indicates no clear connection."
        
        elif chart_type == 'pie':
            explanation = f"This pie chart shows the proportion of your {dataset_name} data that falls into each {x_axis} category. The size of each slice represents the percentage or count, making it easy to see which categories dominate your dataset and which are less common. This helps you understand the overall composition of your data."
        
        elif chart_type == 'histogram':
            explanation = f"This histogram shows the distribution of {x_axis} values in your {dataset_name} dataset. It groups values into ranges (bins) and shows how many data points fall into each range. This reveals whether your data is normally distributed, skewed, or has multiple peaks, helping you understand the typical values and identify outliers."
        
        elif chart_type == 'box':
            if grouping and grouping in self.df.columns:
                explanation = f"This box plot compares the distribution of {y_axis} across different {grouping} categories in your {dataset_name} dataset. Each box shows the median, quartiles, and range of values, making it easy to compare how {y_axis} varies between groups. Outliers are shown as individual points, helping you spot unusual data."
            else:
                explanation = f"This box plot shows the distribution of {y_axis} values in your {dataset_name} dataset. The box contains the middle 50% of data (between the 25th and 75th percentiles), the line inside shows the median, and the whiskers extend to show the full range. Outliers beyond the whiskers are shown as individual points."
        
        else:
            explanation = f"This chart visualizes your {dataset_name} data to help you understand patterns and relationships. It shows {x_axis} on the horizontal axis and helps reveal insights about your data distribution and structure."
        
        return explanation
    
    def _create_chart_from_recommendation(self, viz_rec):
        """Create a specific chart based on AI recommendation"""
        chart_type = viz_rec.get('chart_type', '').lower()
        x_axis = viz_rec.get('x_axis')
        y_axis = viz_rec.get('y_axis')
        title = viz_rec.get('title', 'Data Analysis')
        grouping = viz_rec.get('grouping')
        
        # Ensure columns exist in dataframe
        if x_axis and x_axis not in self.df.columns:
            return None
        if y_axis and y_axis not in self.df.columns:
            return None
        
        try:
            # Clean data for visualization
            viz_df = self.df.copy()
            
            if chart_type == 'bar':
                if y_axis and pd.api.types.is_numeric_dtype(viz_df[y_axis]):
                    # Numeric y-axis: group and aggregate
                    if grouping and grouping in viz_df.columns:
                        grouped_data = viz_df.groupby(x_axis)[y_axis].sum().sort_values(ascending=False).head(15)
                    else:
                        grouped_data = viz_df.groupby(x_axis)[y_axis].sum().sort_values(ascending=False).head(15)
                    
                    # Create more descriptive title
                    if title == 'Data Analysis':
                        title = f"{y_axis} by {x_axis} - Top 15 Categories"
                    
                    fig = px.bar(x=grouped_data.index, y=grouped_data.values, 
                               title=title, labels={'x': x_axis, 'y': y_axis})
                else:
                    # Categorical: show value counts
                    value_counts = viz_df[x_axis].value_counts().head(15)
                    
                    # Create more descriptive title
                    if title == 'Data Analysis':
                        title = f"Distribution of {x_axis} - Top 15 Categories"
                    
                    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                               title=title, labels={'x': x_axis, 'y': 'Count'})
            
            elif chart_type == 'line':
                # Clean data for line chart
                clean_df = viz_df[[x_axis, y_axis]].dropna()
                
                # Create more descriptive title
                if title == 'Data Analysis':
                    title = f"{y_axis} Trend Over {x_axis}"
                
                fig = px.line(clean_df, x=x_axis, y=y_axis, title=title)
            
            elif chart_type == 'scatter':
                # Clean data for scatter plot - ensure numeric columns
                if pd.api.types.is_numeric_dtype(viz_df[x_axis]) and pd.api.types.is_numeric_dtype(viz_df[y_axis]):
                    clean_df = viz_df[[x_axis, y_axis]].dropna()
                    
                    # Create more descriptive title
                    if title == 'Data Analysis':
                        if grouping and grouping in viz_df.columns:
                            title = f"{y_axis} vs {x_axis} (Grouped by {grouping})"
                        else:
                            title = f"{y_axis} vs {x_axis} - Relationship Analysis"
                    
                    if grouping and grouping in viz_df.columns:
                        clean_df = viz_df[[x_axis, y_axis, grouping]].dropna()
                        fig = px.scatter(clean_df, x=x_axis, y=y_axis, color=grouping, title=title)
                    else:
                        fig = px.scatter(clean_df, x=x_axis, y=y_axis, title=title)
                else:
                    # If columns are not numeric, skip this visualization
                    return None
            
            elif chart_type == 'pie':
                value_counts = viz_df[x_axis].value_counts().head(10)
                fig = px.pie(values=value_counts.values, names=value_counts.index, title=title)
            
            elif chart_type == 'histogram':
                # Clean data for histogram
                clean_data = viz_df[x_axis].dropna()
                fig = px.histogram(x=clean_data, title=title, nbins=30, labels={'x': x_axis})
            
            elif chart_type == 'box':
                if grouping and grouping in viz_df.columns:
                    clean_df = viz_df[[grouping, y_axis]].dropna()
                    fig = px.box(clean_df, x=grouping, y=y_axis, title=title)
                else:
                    clean_data = viz_df[y_axis].dropna()
                    fig = px.box(y=clean_data, title=title, labels={'y': y_axis})
            
            else:
                # Default to appropriate chart based on data type
                if pd.api.types.is_numeric_dtype(viz_df[x_axis]):
                    clean_data = viz_df[x_axis].dropna()
                    fig = px.histogram(x=clean_data, title=title, labels={'x': x_axis})
                else:
                    value_counts = viz_df[x_axis].value_counts().head(15)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                               title=title, labels={'x': x_axis, 'y': 'Count'})
            
            # Apply professional styling
            fig.update_layout(
                font=dict(family="Arial, sans-serif", size=12),
                title_font=dict(size=16, family="Arial, sans-serif"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                margin=dict(l=50, r=50, t=80, b=50),
                height=400
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
            
            return fig
            
        except Exception as e:
            print(f"Error creating {chart_type} chart: {str(e)}")
            return None
    
    def _get_fallback_visualizations(self):
        """Fallback visualization recommendations if AI fails"""
        data_summary = self.get_data_summary()
        recommendations = []
        
        numeric_cols = data_summary['numeric_columns']
        categorical_cols = data_summary['categorical_columns']
        
        # Basic visualizations based on data types
        if categorical_cols:
            recommendations.append({
                "chart_type": "bar",
                "x_axis": categorical_cols[0],
                "y_axis": None,
                "title": f"Distribution of {categorical_cols[0]} - Category Analysis",
                "purpose": "Shows frequency distribution of categories"
            })
        
        if numeric_cols:
            recommendations.append({
                "chart_type": "histogram",
                "x_axis": numeric_cols[0],
                "y_axis": None,
                "title": f"Distribution of {numeric_cols[0]} - Value Range Analysis",
                "purpose": "Shows data distribution and identifies patterns"
            })
        
        if len(numeric_cols) >= 2:
            recommendations.append({
                "chart_type": "scatter",
                "x_axis": numeric_cols[0],
                "y_axis": numeric_cols[1],
                "title": f"{numeric_cols[0]} vs {numeric_cols[1]} - Relationship Analysis",
                "purpose": "Shows relationship between variables and identifies correlations"
            })
        
        return recommendations
    
    def _create_basic_visualizations(self):
        """Create basic fallback visualizations"""
        figs = []
        
        try:
            # Clean numeric and categorical columns
            numeric_cols = [col for col in self.df.select_dtypes(include=[np.number]).columns 
                          if not self.df[col].isna().all()]
            categorical_cols = [col for col in self.df.select_dtypes(include=['object']).columns 
                              if not self.df[col].isna().all()]
            
            # Basic histogram for first numeric column
            if len(numeric_cols) > 0:
                clean_data = self.df[numeric_cols[0]].dropna()
                if len(clean_data) > 0:
                    fig1 = px.histogram(x=clean_data, title=f"Distribution of {numeric_cols[0]} - Value Range Analysis", 
                                      labels={'x': numeric_cols[0]}, nbins=30)
                    fig1.update_layout(
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font=dict(size=16, family="Arial, sans-serif"),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=400
                    )
                    explanation = f"This histogram shows the distribution of {numeric_cols[0]} values in your {os.path.basename(self.csv_file_path).replace('.csv', '')} dataset. It groups values into ranges (bins) and shows how many data points fall into each range. This reveals whether your data is normally distributed, skewed, or has multiple peaks, helping you understand the typical values and identify outliers."
                    figs.append(("histogram", fig1.to_html(include_plotlyjs='cdn'), explanation))
            
            # Basic bar chart for first categorical column
            if len(categorical_cols) > 0:
                value_counts = self.df[categorical_cols[0]].value_counts().head(10)
                if len(value_counts) > 0:
                    fig2 = px.bar(x=value_counts.index, y=value_counts.values,
                                 title=f"Top 10 {categorical_cols[0]} Categories - Frequency Analysis",
                                 labels={'x': categorical_cols[0], 'y': 'Count'})
                    fig2.update_layout(
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font=dict(size=16, family="Arial, sans-serif"),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=400
                    )
                    explanation = f"This bar chart displays the frequency count of each {categorical_cols[0]} category in your {os.path.basename(self.csv_file_path).replace('.csv', '')} dataset. It shows how many times each category appears, helping you understand the distribution and identify the most common {categorical_cols[0]} values. This is useful for understanding your data's composition."
                    figs.append(("bar_chart", fig2.to_html(include_plotlyjs='cdn'), explanation))
            
            # Scatter plot for two numeric columns
            if len(numeric_cols) >= 2:
                clean_df = self.df[[numeric_cols[0], numeric_cols[1]]].dropna()
                if len(clean_df) > 0:
                    fig3 = px.scatter(clean_df, x=numeric_cols[0], y=numeric_cols[1],
                                    title=f"{numeric_cols[0]} vs {numeric_cols[1]} - Relationship Analysis")
                    fig3.update_layout(
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font=dict(size=16, family="Arial, sans-serif"),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=400
                    )
                    explanation = f"This scatter plot explores the relationship between {numeric_cols[0]} and {numeric_cols[1]} in your {os.path.basename(self.csv_file_path).replace('.csv', '')} dataset. Each point represents one data record. If points form a clear pattern (like a line going up or down), it suggests a relationship between these variables. Random scatter indicates no clear connection."
                    figs.append(("scatter_plot", fig3.to_html(include_plotlyjs='cdn'), explanation))
        
        except Exception as e:
            print(f"Error creating basic visualizations: {str(e)}")
        
        return figs
    
    def _capture_all_results(self, local_vars):
        """Capture results from all variables in the local namespace"""
        results = {}
        for var_name, var_value in local_vars.items():
            if var_name not in ['df', 'pd', 'np'] and not var_name.startswith('_'):
                # Skip variables with generic names that are likely intermediate results
                if var_name.lower() in ['summary', 'explanation', 'result', 'output', 'data']:
                    continue
                # Skip variables that contain problematic text patterns
                var_str = str(var_value).lower()
                if any(pattern in var_str for pattern in ['converted str to displayable format', 'str format for display purposes']):
                    continue
                try:
                    if isinstance(var_value, pd.DataFrame):
                        # Convert DataFrame to HTML table
                        if len(var_value) > 0:
                            # Limit size for display
                            display_df = var_value.head(10) if len(var_value) > 10 else var_value
                            
                            # Create descriptive name and summary based on block context
                            descriptive_name = self._create_descriptive_name(var_name, "All Results", var_value)
                            descriptive_summary = self._create_descriptive_summary(var_name, "All Results", var_value)
                            
                            # Ensure all required keys are present
                            results[descriptive_name] = {
                                'type': 'dataframe',
                                'shape': var_value.shape,
                                'html': display_df.round(2).to_html(classes='table table-striped table-hover table-sm'),
                                'summary': descriptive_summary,
                                'explanation': self._get_dataframe_explanation("All Results", var_value)
                            }
                    
                    elif isinstance(var_value, pd.Series):
                        # Convert Series to readable format
                        if len(var_value) > 0:
                            display_series = var_value.head(10) if len(var_value) > 10 else var_value
                            
                            # Create descriptive name and summary
                            descriptive_name = self._create_descriptive_name(var_name, "All Results", var_value)
                            descriptive_summary = self._create_descriptive_summary(var_name, "All Results", var_value)
                            
                            # Ensure all required keys are present
                            results[descriptive_name] = {
                                'type': 'series',
                                'length': len(var_value),
                                'html': display_series.round(2).to_frame().to_html(classes='table table-striped table-hover table-sm'),
                                'summary': descriptive_summary,
                                'explanation': self._get_series_explanation("All Results", var_value)
                            }
                    
                    elif isinstance(var_value, (int, float)):
                        # Numeric scalar values
                        descriptive_name = self._create_descriptive_name(var_name, "All Results", var_value)
                        descriptive_summary = self._create_descriptive_summary(var_name, "All Results", var_value)
                        
                        # Ensure all required keys are present
                        results[descriptive_name] = {
                            'type': 'scalar',
                            'value': var_value,
                            'summary': descriptive_summary,
                            'explanation': self._get_scalar_explanation("All Results", var_value)
                        }
                    
                    elif isinstance(var_value, str):
                        # String values - only include if they seem meaningful
                        if len(var_value) > 0 and not var_value.startswith('Converted'):
                            descriptive_name = self._create_descriptive_name(var_name, "All Results", var_value)
                            descriptive_summary = self._create_descriptive_summary(var_name, "All Results", var_value)
                            
                            # Ensure all required keys are present
                            results[descriptive_name] = {
                                'type': 'scalar',
                                'value': var_value,
                                'summary': descriptive_summary,
                                'explanation': self._get_scalar_explanation("All Results", var_value)
                            }
                    
                    elif isinstance(var_value, (list, tuple)):
                        # Handle lists and tuples - only if they contain meaningful data
                        if len(var_value) > 0:
                            # Skip lists that are just column names or intermediate data
                            if not (isinstance(var_value[0], str) and any(col in str(var_value) for col in ['Revenue', 'Company', 'Sector'])):
                                descriptive_name = self._create_descriptive_name(var_name, "All Results", var_value)
                                descriptive_summary = self._create_descriptive_summary(var_name, "All Results", var_value)
                                
                                # Convert to DataFrame for display if possible
                                try:
                                    if isinstance(var_value[0], dict):
                                        # List of dictionaries
                                        temp_df = pd.DataFrame(var_value)
                                        display_data = temp_df.head(10)
                                        results[descriptive_name] = {
                                            'type': 'dataframe',
                                            'shape': temp_df.shape,
                                            'html': display_data.round(2).to_html(classes='table table-striped table-hover table-sm'),
                                            'summary': descriptive_summary,
                                            'explanation': f"List of {len(var_value)} items converted to table format for display."
                                        }
                                    else:
                                        # Simple list
                                        temp_df = pd.DataFrame(var_value, columns=[var_name])
                                        display_data = temp_df.head(10)
                                        results[descriptive_name] = {
                                            'type': 'dataframe',
                                            'shape': temp_df.shape,
                                            'html': display_data.round(2).to_html(classes='table table-striped table-hover table-sm'),
                                            'summary': descriptive_summary,
                                            'explanation': f"List of {len(var_value)} values converted to table format for display."
                                        }
                                except:
                                    # Fallback for complex lists
                                    results[descriptive_name] = {
                                        'type': 'scalar',
                                        'value': f"List with {len(var_value)} items",
                                        'summary': descriptive_summary,
                                        'explanation': f"Complex list structure with {len(var_value)} items that couldn't be converted to table format."
                                    }
                    
                    elif isinstance(var_value, dict):
                        # Handle dictionaries - only if they contain meaningful data
                        if len(var_value) > 0:
                            # Skip dictionaries that are just intermediate data
                            if not any(key in str(var_value) for key in ['dtype', 'Converted', 'str format']):
                                descriptive_name = self._create_descriptive_name(var_name, "All Results", var_value)
                                descriptive_summary = self._create_descriptive_summary(var_name, "All Results", var_value)
                                
                                try:
                                    # Try to convert to DataFrame
                                    temp_df = pd.DataFrame.from_dict(var_value, orient='index')
                                    display_data = temp_df.head(10)
                                    results[descriptive_name] = {
                                        'type': 'dataframe',
                                        'shape': temp_df.shape,
                                        'html': display_data.round(2).to_html(classes='table table-striped table-hover table-sm'),
                                        'summary': descriptive_summary,
                                        'explanation': f"Dictionary converted to table format for display."
                                    }
                                except:
                                    # Fallback for complex dictionaries
                                    results[descriptive_name] = {
                                        'type': 'scalar',
                                        'value': f"Dictionary with {len(var_value)} keys",
                                        'summary': descriptive_summary,
                                        'explanation': f"Complex dictionary structure with {len(var_value)} keys that couldn't be converted to table format."
                                    }
                
                except Exception as e:
                    print(f"⚠️ Warning: Could not process variable {var_name}: {str(e)}")
                    # Create a fallback result to prevent template errors
                    try:
                        fallback_name = self._create_descriptive_name(var_name, "All Results", var_value)
                        fallback_summary = f"Variable {var_name} of type {type(var_value).__name__}"
                        
                        results[fallback_name] = {
                            'type': 'scalar',
                            'value': f"Error processing {var_name}: {str(e)}",
                            'summary': fallback_summary,
                            'explanation': f"This variable could not be processed due to an error: {str(e)}"
                        }
                    except:
                        # Last resort fallback
                        continue
        
        return results
    
    def _create_basic_analysis_results(self):
        """Create a basic set of analysis results if the full execution fails"""
        analysis_results = {}
        
        try:
            # Create a minimal set of results to ensure the dashboard structure
            df_clean = self.df.copy()
            
            # Get actual column names from the dataset
            all_cols = list(df_clean.columns)
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
            
            # Basic statistics for numeric columns
            if numeric_cols:
                df_stats = df_clean[numeric_cols].describe()
                analysis_results['Basic Statistics'] = {
                    'type': 'dataframe',
                    'shape': df_stats.shape,
                    'html': df_stats.round(2).to_html(classes='table table-striped table-hover table-sm'),
                    'summary': f"Basic statistical summary for {len(numeric_cols)} numeric columns.",
                    'explanation': "This table shows key statistical measures (mean, median, standard deviation, etc.) for all numeric columns in your dataset. It helps identify the central tendency and spread of your data."
                }
            
            # Basic groupby if categorical columns exist
            if categorical_cols and numeric_cols:
                for cat_col in categorical_cols[:2]: # Limit to first 2 categorical columns
                    for num_col in numeric_cols[:2]: # Limit to first 2 numeric columns
                        try:
                            group_result = df_clean.groupby(cat_col)[num_col].mean()
                            if len(group_result) > 0:
                                analysis_results[f'Group Analysis: {cat_col} by {num_col}'] = {
                                    'type': 'series',
                                    'length': len(group_result),
                                    'html': group_result.round(2).to_frame().to_html(classes='table table-striped table-hover table-sm'),
                                    'summary': f"Mean {num_col} by {cat_col}.",
                                    'explanation': f"This table shows the average {num_col} for each {cat_col} category."
                                }
                        except:
                            continue
            
            # Top values for key columns
            if numeric_cols:
                for col in numeric_cols[:2]: # Limit to first 2 numeric columns
                    try:
                        # Use available columns for display
                        display_cols = [col]
                        if 'Company' in all_cols:
                            display_cols.insert(0, 'Company')
                        elif categorical_cols:
                            display_cols.insert(0, categorical_cols[0])
                        
                        top_values = df_clean.nlargest(10, col)[display_cols]
                        analysis_results[f'Top Values: {col}'] = {
                            'type': 'dataframe',
                            'shape': top_values.shape,
                            'html': top_values.round(2).to_html(classes='table table-striped table-hover table-sm'),
                            'summary': f"Top 10 values for {col}.",
                            'explanation': f"This table shows the 10 highest values for the '{col}' column."
                        }
                    except:
                        continue
            
            # Basic correlation analysis if numeric columns exist
            if len(numeric_cols) > 1:
                try:
                    corr_data = df_clean[numeric_cols].apply(pd.to_numeric, errors='coerce').dropna()
                    if len(corr_data) > 0:
                        corr_matrix = corr_data.corr()
                        analysis_results['Correlation Analysis'] = {
                            'type': 'dataframe',
                            'shape': corr_matrix.shape,
                            'html': corr_matrix.round(3).to_html(classes='table table-striped table-hover table-sm'),
                            'summary': f"Correlation matrix for {len(numeric_cols)} numeric columns.",
                            'explanation': "This correlation matrix shows how strongly different numeric variables are related to each other. Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation)."
                        }
                except:
                    pass
            
        except Exception as e:
            print(f"  ❌ Error creating basic analysis results: {str(e)}")
        
        return analysis_results
    
    def generate_dashboard_html(self, visualizations, analysis_results):
        """Generate professional HTML dashboard using Jinja2 template"""
        
        # Professional HTML template with analysis results
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ dataset_name }} - Data Analysis Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --accent-color: #e74c3c;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --light-bg: #f8f9fa;
            --card-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
            --border-radius: 8px;
        }
        
        body {
            background-color: #f5f7fa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: var(--primary-color);
        }
        
        .dashboard-header {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            padding: 3rem 0 2rem 0;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }
        
        .dashboard-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="white" stroke-width="0.5" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
        }
        
        .dashboard-header .container {
            position: relative;
            z-index: 1;
        }
        
        .dashboard-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .dashboard-subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
            margin-bottom: 1rem;
        }
        

        
        .viz-container {
            background: white;
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            margin-bottom: 2rem;
            padding: 0;
            border: 1px solid rgba(0,0,0,0.04);
            overflow: hidden;
        }
        
        .viz-container .alert {
            margin: 0;
            border-radius: 0;
            border: none;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-bottom: 1px solid #dee2e6;
        }
        
        .viz-content {
            padding: 1rem;
        }
        
        .analysis-section {
            background: white;
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            margin-bottom: 2rem;
            border: 1px solid rgba(0,0,0,0.04);
        }
        
        .analysis-header {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 1.5rem;
            border-bottom: 1px solid #dee2e6;
            border-radius: var(--border-radius) var(--border-radius) 0 0;
        }
        
        .analysis-content {
            padding: 1.5rem;
        }
        
        .insights-card {
            background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
            border-radius: var(--border-radius);
            padding: 2rem;
            margin-bottom: 2rem;
            border-left: 5px solid var(--secondary-color);
            box-shadow: var(--card-shadow);
        }
        
        .insights-content {
            line-height: 1.7;
            font-size: 1.05rem;
        }
        
        .insights-content ul {
            margin: 1rem 0;
            padding-left: 1.5rem;
        }
        
        .insights-content li {
            margin-bottom: 0.75rem;
            color: var(--primary-color);
        }
        
        .alert {
            border-radius: var(--border-radius);
            border: none;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        .alert-info {
            background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
            border-left: 4px solid var(--secondary-color);
            color: var(--primary-color);
        }
        
        .alert-success {
            background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
            border-left: 4px solid var(--success-color);
            color: var(--primary-color);
        }
        
        .alert i {
            opacity: 0.8;
        }
        
        .section-title {
            font-size: 1.75rem;
            font-weight: 600;
            color: var(--primary-color);
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .section-title i {
            color: var(--secondary-color);
        }
        
        .analysis-block {
            margin-bottom: 2rem;
            padding: 1.5rem;
            background: #f8f9fa;
            border-radius: var(--border-radius);
            border-left: 4px solid var(--warning-color);
            position: relative;
        }
        
        .analysis-block .alert {
            border-radius: var(--border-radius);
            border: none;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .analysis-block .alert-info {
            background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
            border-left: 4px solid var(--secondary-color);
        }
        
        .analysis-block .alert-success {
            background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
            border-left: 4px solid var(--success-color);
        }
        
        .analysis-block h4 {
            color: var(--primary-color);
            margin-bottom: 1rem;
            font-weight: 600;
        }
        
        .analysis-table {
            margin-bottom: 0;
            font-size: 0.9rem;
        }
        
        .analysis-table thead th {
            background-color: var(--secondary-color);
            color: white;
            border: none;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.8rem;
            padding: 0.75rem 0.5rem;
        }
        
        .analysis-table tbody tr:hover {
            background-color: rgba(52, 152, 219, 0.05);
        }
        
        .analysis-table tbody td {
            padding: 0.75rem 0.5rem;
            border-color: #eee;
        }
        
        .data-table-container {
            background: white;
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            border: 1px solid rgba(0,0,0,0.04);
            overflow: hidden;
        }
        
        .table {
            margin-bottom: 0;
        }
        
        .table thead th {
            background-color: var(--primary-color);
            color: white;
            border: none;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.85rem;
            padding: 1rem 0.75rem;
        }
        
        .table tbody tr:hover {
            background-color: rgba(52, 152, 219, 0.05);
        }
        
        .table tbody td {
            padding: 0.875rem 0.75rem;
            border-color: #eee;
        }
        
        .footer {
            margin-top: 4rem;
            padding: 2rem 0;
            text-align: center;
            color: #6c757d;
            border-top: 1px solid #eee;
            background: white;
        }
        
        @media (max-width: 768px) {
            .dashboard-title {
                font-size: 2rem;
            }
            
            .analysis-content {
                padding: 1rem;
            }
        }
    </style>
</head>
<body>
    
    <div class="dashboard-header">
        <div class="container">
            <div class="row">
                <div class="col-lg-8">
                    <h1 class="dashboard-title">
                        <i class="fas fa-chart-line me-3"></i>{{ dataset_name }} Analysis
                    </h1>
                    <p class="dashboard-subtitle">Data Insights & Visualizations</p>
                    <p class="mb-0">
                        <i class="fas fa-calendar-alt me-2"></i>Generated on {{ timestamp }}
                    </p>
                </div>
            </div>
        </div>
    </div>
    
    <div class="container">
        <!-- Dataset Introduction -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="alert alert-info">
                    <i class="fas fa-info-circle me-2"></i>
                    <strong>What We're Analyzing:</strong> This dashboard analyzes your <strong>{{ dataset_name }}</strong> dataset to uncover meaningful patterns, trends, and insights that can help you make better decisions. Each section below provides a different perspective on your data.
                </div>
            </div>
        </div>
        
        <!-- AI-Generated Insights -->
        {% if analysis_insights %}
        <div class="row">
            <div class="col-12">
                <div class="insights-card">
                    <h2 class="section-title mb-3">
                        <i class="fas fa-lightbulb"></i>What Your Data Reveals
                    </h2>
                    <div class="alert alert-success mb-3">
                        <i class="fas fa-robot me-2"></i>
                        <strong>Key Findings:</strong> Based on our analysis of your {{ dataset_name }} data, here are the most important patterns and insights we discovered. These findings can help you understand trends, identify opportunities, and make data-driven decisions.
                    </div>
                    <div class="insights-content">
                        {{ analysis_insights | safe }}
                    </div>
                </div>
            </div>
        </div>
        {% endif %}
        
        <!-- Analysis Results -->
        {% if analysis_results %}
        <div class="row">
            <div class="col-12">
                <h2 class="section-title">
                    <i class="fas fa-microscope"></i>Detailed Analysis
                </h2>
                <div class="alert alert-info mb-4">
                    <i class="fas fa-info-circle me-2"></i>
                    <strong>Understanding Your Results:</strong> Below you'll find the detailed analysis of your {{ dataset_name }} data. Each section shows different calculations and insights that reveal patterns, trends, and relationships in your data. The explanations help you understand what each result means for your business or research.
                </div>
            </div>
        </div>
        
        {% for block_name, block_results in analysis_results.items() %}
        <div class="row">
            <div class="col-12">
                <div class="analysis-section">
                    <div class="analysis-header">
                        <h3 class="mb-0">{{ block_name.replace('_', ' ').replace('analysis', 'Analysis').replace('data', 'Data').title() }}</h3>
                    </div>
                    <div class="analysis-content">
                        {% for var_name, var_data in block_results.items() %}
                        {% if var_name not in ['type', 'shape', 'html', 'length', 'value'] %}
                        <div class="analysis-block">
                            <h4>{{ var_name }}</h4>
                            <p class="text-muted mb-3">{{ var_data.summary }}</p>
                            {% if var_data.explanation %}
                            <div class="alert alert-info mb-3">
                                <i class="fas fa-info-circle me-2"></i>
                                <strong>What This Shows:</strong> {{ var_data.explanation }}
                            </div>
                            {% endif %}
                            {% if var_data.type in ['dataframe', 'series'] %}
                            <div class="table-responsive">
                                {{ var_data.html | safe }}
                            </div>
                            {% elif var_data.type == 'scalar' %}
                            <div class="alert alert-success">
                                <i class="fas fa-calculator me-2"></i>
                                <strong>Calculated Result:</strong> {{ var_data.value }}
                            </div>
                            {% endif %}
                        </div>
                        {% endif %}
                        {% endfor %}
                    </div>
                </div>
            </div>
        </div>
        {% endfor %}
        {% endif %}
        
        <!-- Visualizations -->
        <div class="row">
            <div class="col-12">
                <h2 class="section-title">
                    <i class="fas fa-chart-pie"></i>Charts & Graphs
                </h2>
                <div class="alert alert-info mb-4">
                    <i class="fas fa-info-circle me-2"></i>
                    <strong>Visualizing Your Data:</strong> Charts and graphs make it easier to spot patterns and trends in your {{ dataset_name }} data. Each visualization below shows your data from a different angle, helping you quickly identify insights that might be hidden in tables of numbers.
                </div>
            </div>
        </div>
        
        {% for viz_name, viz_html, viz_explanation in visualizations %}
        <div class="row">
            <div class="col-12">
                <div class="viz-container">
                    <div class="viz-content">
                        <div class="alert alert-info mb-3">
                            <i class="fas fa-info-circle me-2"></i>
                            <strong>What This Chart Shows:</strong> {{ viz_explanation }}
                        </div>
                        {{ viz_html | safe }}
                    </div>
                </div>
            </div>
        </div>
        {% endfor %}
        
        <!-- Data Sample -->
        <div class="row">
            <div class="col-12">
                <h2 class="section-title">
                    <i class="fas fa-table"></i>Your Data Preview
                </h2>
                <div class="alert alert-info mb-3">
                    <i class="fas fa-info-circle me-2"></i>
                    <strong>Raw Data Sample:</strong> Here's a preview of your {{ dataset_name }} data showing the first 10 rows. This gives you a quick look at the actual values and structure of your dataset to understand what we're working with.
                </div>
                <div class="data-table-container">
                    <div class="table-responsive">
                        {{ data_table | safe }}
                    </div>
                </div>
            </div>
        </div>
        
    </div>
    
    <div class="footer">
        <div class="container">
            <p class="mb-0">
                <i class="fas fa-robot me-2"></i>Dashboard generated by AI analysis of your {{ dataset_name }} data
            </p>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
        """
        
        # Generate AI insights from analysis results
        analysis_insights = ""
        if analysis_results:
            analysis_insights = self.generate_analysis_insights(analysis_results)
            # Convert insights to HTML format if it contains bullet points
            if "•" in analysis_insights or "*" in analysis_insights:
                # Convert markdown-style bullet points to HTML
                lines = analysis_insights.split('\n')
                formatted_lines = []
                in_list = False
                for line in lines:
                    line = line.strip()
                    if line.startswith('•') or line.startswith('*') or line.startswith('-'):
                        if not in_list:
                            formatted_lines.append('<ul>')
                            in_list = True
                        formatted_lines.append(f'<li>{line[1:].strip()}</li>')
                    else:
                        if in_list:
                            formatted_lines.append('</ul>')
                            in_list = False
                        if line:
                            formatted_lines.append(f'<p>{line}</p>')
                if in_list:
                    formatted_lines.append('</ul>')
                analysis_insights = '\n'.join(formatted_lines)
        
        # Prepare template data
        template_data = {
            'dataset_name': os.path.basename(self.csv_file_path).replace('.csv', '').replace('_', ' ').title(),
            'timestamp': datetime.now().strftime('%B %d, %Y at %I:%M %p'),
            'visualizations': visualizations,
            'analysis_results': analysis_results,
            'analysis_insights': analysis_insights,
            'data_table': self.df.head(10).to_html(classes='table table-striped table-hover', table_id='data-preview')
        }
        
        # Render template
        template = Template(template_str)
        html_content = template.render(**template_data)
        
        return html_content
    
    def run_complete_analysis(self, output_file='dashboard.html'):
        """Run the complete analysis pipeline"""
        print("🚀 Starting CSV Dashboard Generation Pipeline...")
        
        # Step 1: Load and analyze CSV
        if not self.load_and_analyze_csv():
            return False
        
        # Step 2: Get data summary
        print("📊 Generating data summary...")
        data_summary = self.get_data_summary()
        
        # Step 3: Generate dashboard recommendations (for internal use only)
        print("🤖 Generating AI dashboard recommendations...")
        self.generate_dashboard_recommendations(data_summary)
        print("\n" + "="*80)
        print("AI DASHBOARD RECOMMENDATIONS:")
        print("="*80)
        print(self.insights)
        print("="*80 + "\n")
        
        # Step 4: Generate analysis operations
        print("⚙️ Generating pandas analysis operations...")
        self.generate_analysis_operations(data_summary)
        print("\n" + "-"*60)
        print("GENERATED PANDAS OPERATIONS:")
        print("-"*60)
        print(self.analysis_operations)
        print("-"*60 + "\n")
        
        # Step 5: Execute analysis operations and capture results
        print("🔄 Executing analysis operations...")
        analysis_results = self.execute_analysis_operations()
        
        # Validate and clean analysis results to prevent template errors
        print("🔍 Validating analysis results...")
        analysis_results = self._validate_analysis_results(analysis_results)
        
        # Step 6: Create smart visualizations
        print("📈 Creating intelligent visualizations...")
        visualizations = self.create_visualizations()
        
        # Step 7: Generate professional HTML dashboard with analysis results
        print("🎨 Generating professional HTML dashboard...")
        html_content = self.generate_dashboard_html(visualizations, analysis_results)
        
        # Step 8: Save dashboard
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Professional dashboard generated successfully: {output_file}")
        print(f"📂 Open {output_file} in your browser to view the dashboard!")
        print(f"📊 Dashboard includes:")
        print(f"   • {len(visualizations)} intelligent visualizations")
        print(f"   • {len(analysis_results)} analysis result blocks")
        print(f"   • AI-generated business insights")
        print(f"   • Professional styling and layout")
        
        return True
    
    def _validate_analysis_results(self, analysis_results):
        """Validate and clean analysis results to ensure they have the expected structure"""
        if not analysis_results:
            return {}
        
        validated_results = {}
        
        for block_name, block_results in analysis_results.items():
            # Ensure block_results is a dictionary
            if not isinstance(block_results, dict):
                print(f"⚠️ Warning: Skipping non-dict block_results for {block_name}")
                continue
            
            validated_block = {}
            for var_name, var_data in block_results.items():
                # Ensure var_data is a dictionary
                if not isinstance(var_data, dict):
                    print(f"⚠️ Warning: Skipping non-dict var_data for {var_name} in {block_name}")
                    continue
                
                # Check if this is already a properly formatted result
                if 'type' in var_data and 'summary' in var_data and 'explanation' in var_data:
                    # This result is already properly formatted, just ensure type-specific keys
                    if var_data['type'] == 'dataframe':
                        if 'shape' not in var_data:
                            var_data['shape'] = (0, 0)
                        if 'html' not in var_data:
                            var_data['html'] = '<p>No data available</p>'
                    elif var_data['type'] == 'series':
                        if 'length' not in var_data:
                            var_data['length'] = 0
                        if 'html' not in var_data:
                            var_data['html'] = '<p>No data available</p>'
                    elif var_data['type'] == 'scalar':
                        if 'value' not in var_data:
                            var_data['value'] = 'No value available'
                    
                    # Use the result as-is since it's already properly formatted
                    validated_block[var_name] = var_data
                    continue
                
                # Only add missing keys if the result doesn't have the basic structure
                required_keys = ['type', 'summary', 'explanation']
                missing_keys = [key for key in required_keys if key not in var_data]
                
                if missing_keys:
                    print(f"⚠️ Warning: Adding missing keys {missing_keys} for {var_name} in {block_name}")
                    # Add missing keys with default values
                    for key in missing_keys:
                        if key == 'type':
                            var_data[key] = 'scalar'  # Default type
                        elif key == 'summary':
                            var_data[key] = f"Result for {var_name}"
                        elif key == 'explanation':
                            var_data[key] = f"This result shows data for {var_name}"
                
                # Ensure type-specific keys are present
                if var_data['type'] == 'dataframe':
                    if 'shape' not in var_data:
                        var_data['shape'] = (0, 0)
                    if 'html' not in var_data:
                        var_data['html'] = '<p>No data available</p>'
                elif var_data['type'] == 'series':
                    if 'length' not in var_data:
                        var_data['length'] = 0
                    if 'html' not in var_data:
                        var_data['html'] = '<p>No data available</p>'
                elif var_data['type'] == 'scalar':
                    if 'value' not in var_data:
                        var_data['value'] = 'No value available'
                
                validated_block[var_name] = var_data
            
            if validated_block:  # Only add if we have valid data
                validated_results[block_name] = validated_block
        
        return validated_results
    
    def _test_corrected_code(self):
        """Test if the corrected code is valid before execution"""
        try:
            # Basic syntax check
            compile(self.analysis_operations, '<string>', 'exec')
            
            # Check for common issues
            if 'df_clean' in self.analysis_operations:
                print("  ⚠️ Warning: Code still contains 'df_clean' references")
                return False
                
            if 'undefined_variable' in self.analysis_operations:
                print("  ⚠️ Warning: Code contains placeholder text")
                return False
                
            print("  ✅ Corrected code validation passed")
            return True
            
        except SyntaxError as e:
            print(f"  ❌ Corrected code has syntax errors: {str(e)}")
            return False
        except Exception as e:
            print(f"  ❌ Code validation failed: {str(e)}")
            return False

