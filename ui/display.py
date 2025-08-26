import streamlit as st
from .components import UIComponents
from utils import truncate_description
from config.topics import TOPIC_QUERIES
from embeddings import EmbeddingManager

class ResultDisplay:
    def __init__(self):
        self.ui = UIComponents()
        self.embedding_manager = EmbeddingManager()
    
    def display_low_relevance(self):
        """Display message for low relevance queries."""
        container = st.empty()
        self.ui.stream_text(container, 
            "Your question may not be answerable with PLIDA or lacks specificity. Try again.")
    
    def display_dataset_results(self, user_input, dataset_results, topic, datasets_df, search_engine):
        """Display relevant datasets."""
        st.subheader("Most Relevant Datasets")
        container = st.empty()
        self.ui.stream_text(container, f"**Research Question:** {user_input}")
        
        relevant_datasets = []
        
        # Display main results
        for result in dataset_results:
            row = result['row']
            score = result['score']
            
            # Skip ACLD for healthcare
            if not (topic == "healthcare" and row["dataset_id"].upper() == "ACLD"):
                truncated_desc = truncate_description(row['dataset_description'])
                dataset_text = (
                    f"- **Dataset:** {row['dataset_id']} | "
                    f"**Name:** {row['dataset_name']} | "
                    f"**Description:** {truncated_desc} | "
                    f"**Score:** {score:.3f}"
                )
                dataset_container = st.empty()
                self.ui.stream_text(dataset_container, dataset_text)
                relevant_datasets.append(row['dataset_name'])
        
        # Suggest DOMINO for specific topics
        if topic in ["unemployment", "social services"]:
            self._suggest_domino(datasets_df, relevant_datasets, topic)
        
        # Topic-specific search
        self._display_topic_datasets(topic, datasets_df, search_engine, relevant_datasets)
        
        return relevant_datasets
    
    def _suggest_domino(self, datasets_df, relevant_datasets, topic):
        """Suggest DOMINO dataset for relevant topics."""
        domino_row = datasets_df[datasets_df["dataset_id"].str.upper() == "DOMINO"]
        if not domino_row.empty:
            domino_row = domino_row.iloc[0]
            if domino_row['dataset_name'] not in relevant_datasets:
                truncated_desc = truncate_description(domino_row['dataset_description'])
                dataset_text = (
                    f"- **Dataset ID:** {domino_row['dataset_id']} | "
                    f"**Name:** {domino_row['dataset_name']} | "
                    f"**Description:** {truncated_desc} | "
                    f"**Score:** Suggested for {topic}"
                )
                dataset_container = st.empty()
                self.ui.stream_text(dataset_container, dataset_text)
                relevant_datasets.append(domino_row['dataset_name'])
    
    def _display_topic_datasets(self, topic, datasets_df, search_engine, relevant_datasets):
        """Display topic-specific dataset suggestions."""
        topic_query = TOPIC_QUERIES.get(topic, "")
        if topic_query:
            topic_results = search_engine.search_datasets(topic_query)
            
            topic_container = st.empty()
            self.ui.stream_text(topic_container, f"{topic.capitalize()}-related datasets you might consider:")
            
            for result in topic_results:
                row = result['row']
                if not (topic == "healthcare" and row["dataset_id"].upper() == "ACLD") and \
                   row['dataset_name'] not in relevant_datasets:
                    relevant_datasets.append(row['dataset_name'])
            
            updated_list_container = st.empty()
            self.ui.stream_text(updated_list_container, 
                              "\n".join([f"- {name}" for name in relevant_datasets]))
    
    def display_variable_results(self, variable_desc, index, search_results, dataset_note):
        """Display variable search results."""
        
        # Collect all dataset IDs from results for linking analysis
        dataset_ids = set()
        for result in search_results:
            dataset_id = result['row'].get('dataset_id', '')
            if dataset_id:
                dataset_ids.add(dataset_id)
        
        # Separate regular results from CORE alternatives
        regular_results = [r for r in search_results if not r.get('is_core_alternative', False)]
        core_alternatives = [r for r in search_results if r.get('is_core_alternative', False)]
        
        # Display regular results first
        if regular_results:
            match_container = st.empty()
            self.ui.stream_text(match_container, f"**Matching PLIDA Variables{dataset_note}:**")
            
            for result in regular_results:
                self._display_single_variable_result(result)
        
        # Display CORE alternatives if any
        if core_alternatives:
            st.markdown("---")
            core_container = st.empty()
            self.ui.stream_text(core_container, "**🔄 CORE Demographic Alternatives** *(Better population alignment)*:")
            
            for result in core_alternatives:
                self._display_single_variable_result(result, is_core_alternative=True)
        
        return dataset_ids
    
    def _display_single_variable_result(self, result, is_core_alternative=False):
        """Display a single variable result."""
        row = result['row']
        score = result['score']
        truncated_desc = truncate_description(row['description'])
        
        # Add indicator if this result was boosted
        boost_indicator = " ⭐" if result.get('boosted', False) else ""
        
        # Add indicator for CORE alternatives
        core_indicator = " 🔄" if is_core_alternative else ""
        
        # Check for population match information
        population_match = result.get('population_match')
        population_warning = ""
        
        if population_match:
            if population_match['match'] == 'maybe':
                population_warning = " ⚠️ **Population Check Required**"
            elif population_match['match'] == 'no':
                population_warning = " ❌ **Population Mismatch**"
            elif population_match['match'] == 'error':
                population_warning = " ⚠️ **Population Check Error**"
            elif population_match['match'] == 'yes' and is_core_alternative:
                population_warning = " ✅ **Good Population Match**"
        
        match_text = (
            f"- **Dataset:** {row['dataset']}{boost_indicator}{core_indicator} | "
            f"**Variable:** {row['variable_name']} | "
            f"**Description:** {truncated_desc} | "
            f"**Score:** {score:.3f}{population_warning}"
        )
        match_container = st.empty()
        self.ui.stream_text(match_container, match_text)
        
        # Show population match details if available
        if population_match and population_match['match'] in ['maybe', 'error']:
            with st.expander(f"Population Check Details - {row['variable_name']}", expanded=False):
                st.caption(f"**Assessment:** {population_match['match'].title()}")
                st.caption(f"**Reasoning:** {population_match['reasoning']}")
                if population_match.get('confidence'):
                    st.caption(f"**Confidence:** {population_match['confidence']:.0%}")
                
                if population_match['match'] == 'maybe':
                    st.warning("⚠️ **Action Required:** Please manually verify if this dataset's population aligns with your research question before using this variable.")
                elif population_match['match'] == 'error':
                    st.error("❌ **Error:** Unable to assess population alignment. Please check manually.")
        
        # Show ACLD notice if this is an ACLD variable
        if row.get('dataset_id') == 'ACLD':
            with st.expander(f"📊 Census Data Alternative - {row['variable_name']}", expanded=False):
                st.info(
                    "**Alternative Available:** The full Census of Population and Housing should have the same variables. "
                    "Consider what you value more:\n"
                    "• **Cross-sectional representation** (full Census) - Complete population coverage\n"
                    "• **Linkage of individuals between Census years** (ACLD) - Longitudinal tracking with 5% sample"
                )
                st.caption("ACLD provides longitudinal linkage but covers only a 5% sample of the population, while the full Census provides complete population coverage for cross-sectional analysis.")
    
    def check_and_display_linking_strategy(self, dataset_ids_collection):
        """Check for SYNTHETIC_AEUID in datasets and display linking strategy."""
        import pandas as pd
        
        if not dataset_ids_collection:
            return
        
        # Flatten the collection of dataset ID sets into a single set
        all_dataset_ids = set()
        for dataset_ids in dataset_ids_collection:
            all_dataset_ids.update(dataset_ids)
        
        if not all_dataset_ids:
            return
            
        try:
            # Load plida4.csv to check for SYNTHETIC_AEUID
            plida4_df = pd.read_csv('resources/plida4.csv')
            
            # Find datasets that contain SYNTHETIC_AEUID
            synthetic_datasets = plida4_df[
                (plida4_df['dataset_id'].isin(all_dataset_ids)) &
                (plida4_df['variable_name'] == 'SYNTHETIC_AEUID')
            ]['dataset_id'].unique().tolist()
            
            if synthetic_datasets:
                st.markdown("---")
                st.markdown("## 🔗 Linking Strategy")
                
                st.info(
                    "The following datasets contain the **SYNTHETIC_AEUID** variable, which enables potential "
                    "data linking capabilities:"
                )
                
                # Display linkable datasets
                for dataset_id in sorted(synthetic_datasets):
                    # Get dataset name from plida4
                    dataset_info = plida4_df[plida4_df['dataset_id'] == dataset_id].iloc[0]
                    dataset_name = dataset_info['dataset']
                    st.write(f"• **{dataset_id}**: {dataset_name}")
                
                st.warning(
                    "⚠️ **Important:** While these datasets have linking potential through SYNTHETIC_AEUID, "
                    "you must verify the specific linking rules, restrictions, and documentation before "
                    "attempting to link datasets. Contact the PLIDA team for detailed linking guidelines."
                )
                
        except Exception as e:
            st.error(f"Error checking linking strategy: {e}")
    
    def display_debug_info(self, gpt_data):
        """Display debug information."""
        with st.expander("Raw GPT-4o Response (Debug)"):
            st.write(gpt_data.get('raw_response', ''))
        
        with st.expander("Parsed JSON Response"):
            st.json({k: v for k, v in gpt_data.items() if k != 'raw_response'})
        
        # Display medical condition detection debug info if available
        if gpt_data.get('medical_condition_detected'):
            with st.expander("Medical Condition Detection (Debug)"):
                medical_info = gpt_data['medical_condition_detected']
                st.json({k: v for k, v in medical_info.items() if k != 'raw_response'})
                if medical_info.get('raw_response'):
                    st.text("Raw Medical Detection Response:")
                    st.write(medical_info['raw_response'])
        
        # Display geographic analysis detection debug info if available
        if gpt_data.get('geographic_analysis_detected'):
            with st.expander("Geographic Analysis Detection (Debug)"):
                geographic_info = gpt_data['geographic_analysis_detected']
                st.json({k: v for k, v in geographic_info.items() if k != 'raw_response'})
                if geographic_info.get('raw_response'):
                    st.text("Raw Geographic Detection Response:")
                    st.write(geographic_info['raw_response'])
    
    def display_causal_variables(self, categorized, search_engine, search_filters, topic, ui, relevant_datasets=None, user_input=None, query_analyzer=None):
        """Display variables categorized for causal analysis.
        
        Args:
            categorized: Categorized variables
            search_engine: Search engine instance
            search_filters: Search filters instance
            topic: Current topic
            ui: UI components instance
            relevant_datasets: List of selected dataset names to prioritize
        """
        st.subheader("🔬 Variables for Causal Analysis")
        
        # Show prioritization note if relevant datasets exist
        if relevant_datasets:
            st.caption("⭐ *Variables from the most relevant datasets are prioritized in the results*")
        
        # Display Potential Causal Variables
        if categorized['causal_variables']:
            st.markdown("### 🎯 Potential Causal Variables (Treatment/Exposure)")
            st.caption("*Variables that might cause or influence the outcome*")
            self._display_variable_section(
                categorized['causal_variables'], 
                search_engine, 
                search_filters, 
                topic, 
                ui,
                "causal",
                relevant_datasets,
                user_input,
                query_analyzer
            )
        
        # Display Outcome Variables
        if categorized['outcome_variables']:
            st.markdown("### 📊 Outcome Variables (Dependent)")
            st.caption("*Variables being affected or measured as results*")
            self._display_variable_section(
                categorized['outcome_variables'], 
                search_engine, 
                search_filters, 
                topic, 
                ui,
                "outcome",
                relevant_datasets,
                user_input,
                query_analyzer
            )
        
        # Display Control Variables
        if categorized['control_variables']:
            st.markdown("### 🔧 Control Variables (Confounders/Covariates)")
            st.caption("*Variables that need to be controlled for valid causal inference*")
            self._display_variable_section(
                categorized['control_variables'], 
                search_engine, 
                search_filters, 
                topic, 
                ui,
                "control",
                relevant_datasets,
                user_input,
                query_analyzer
            )
    
    def _display_variable_section(self, variables, search_engine, search_filters, topic, ui, category, relevant_datasets=None, user_input=None, query_analyzer=None):
        """Display a section of categorized variables.
        
        Args:
            variables: List of variable descriptions
            search_engine: Search engine instance
            search_filters: Search filters instance
            topic: Current topic
            ui: UI components instance
            category: Variable category (causal, outcome, control, etc.)
            relevant_datasets: List of selected dataset names to prioritize
        """
        # Collect dataset IDs for linking strategy
        all_dataset_ids = []
        for i, var_desc in enumerate(variables):
            with st.container():
                # Display variable description
                container = st.empty()
                ui.stream_text(container, f"**{i+1}. {var_desc}**")
                
                # Determine appropriate index
                index_name, dataset_note, additional_searches = search_filters.get_appropriate_index(
                    var_desc, topic
                )
                
                # Enhance query for employment variables
                search_query = var_desc
                if search_filters.is_employment_variable(var_desc):
                    search_query += " jobseeker"
                
                # Search for matching variables with dataset boost
                var_results = search_engine.search_variables(search_query, index_name, boost_datasets=relevant_datasets)
                var_results = search_engine.deduplicate_results(var_results)
                
                # Apply population filtering if available
                if user_input and query_analyzer and var_results:
                    with st.spinner("🔍 Checking population alignment..."):
                        # Use the specific variable description for population matching instead of original user query
                        var_results = query_analyzer.check_population_match(var_desc, var_results, search_engine)
                
                # Display results and collect dataset IDs
                if var_results:
                    dataset_ids = self.display_variable_results(var_desc, i, var_results, dataset_note)
                    if dataset_ids:
                        all_dataset_ids.append(dataset_ids)
                else:
                    st.caption("No matching PLIDA variables found")
                
                # Perform additional searches for employment variables
                for additional in additional_searches:
                    enhanced_query = var_desc + additional['query_suffix']
                    add_results = search_engine.search_variables(enhanced_query, additional['index'], boost_datasets=relevant_datasets)
                    add_results = search_engine.deduplicate_results(add_results)
                    
                    # Apply population filtering to additional results
                    if user_input and query_analyzer and add_results:
                        # Use the enhanced query for additional searches instead of original user query
                        add_results = query_analyzer.check_population_match(enhanced_query, add_results, search_engine)
                    
                    if add_results:
                        dataset_ids = self.display_variable_results(
                            var_desc, i, add_results, additional['note']
                        )
                        if dataset_ids:
                            all_dataset_ids.append(dataset_ids)
                
                st.write("---")
        
        # Display linking strategy at the end of the section
        if all_dataset_ids:
            self.check_and_display_linking_strategy(all_dataset_ids)
    
    def display_descriptive_variables(self, categorized, search_engine, search_filters, topic, ui, relevant_datasets=None, user_input=None, query_analyzer=None):
        """Display variables categorized for descriptive analysis.
        
        Args:
            categorized: Categorized variables
            search_engine: Search engine instance
            search_filters: Search filters instance
            topic: Current topic
            ui: UI components instance
            relevant_datasets: List of selected dataset names to prioritize
        """
        st.subheader("📊 Variables for Descriptive Analysis")
        
        # Show prioritization note if relevant datasets exist
        if relevant_datasets:
            st.caption("⭐ *Variables from the most relevant datasets are prioritized in the results*")
        
        # Display Main Variables
        if categorized['main_variables']:
            st.markdown("### 🎯 Main Variables")
            st.caption("*Primary variables that directly address the research question*")
            self._display_variable_section(
                categorized['main_variables'],
                search_engine,
                search_filters,
                topic,
                ui,
                "main",
                relevant_datasets,
                user_input,
                query_analyzer
            )
        
        # Display Subgroups
        if categorized['subgroups']:
            st.markdown("### 👥 Subgroups")
            st.caption("*Variables used to break down or segment the analysis*")
            self._display_variable_section(
                categorized['subgroups'],
                search_engine,
                search_filters,
                topic,
                ui,
                "subgroup",
                relevant_datasets,
                user_input,
                query_analyzer
            )
    
    def display_medical_condition_suggestion(self, medical_info, datasets_df, relevant_datasets):
        """Display PBS dataset suggestion for medical condition queries."""
        # Show medical detection info even if suggest_pbs is False
        if medical_info.get('is_medical', False):
            st.success("🏥 **Medical Condition Detected**")
            st.caption(f"*Confidence: {medical_info.get('confidence', 0):.0%}*")
            
            if medical_info.get('medical_keywords'):
                keywords_text = ", ".join(medical_info['medical_keywords'])
                st.caption(f"*Medical terms identified: {keywords_text}*")
            
            # Always show PBS recommendation prominently
            st.markdown("---")
            st.markdown("## 💊 **PBS Dataset Recommendation**")
            st.info("For medical condition queries, we strongly recommend using the **PBS dataset** with the **DRG_TYPE_CD variable**.")
            
            # Find PBS dataset
            pbs_row = datasets_df[datasets_df["dataset_id"].str.upper() == "PBS"]
            if not pbs_row.empty:
                pbs_row = pbs_row.iloc[0]
                if pbs_row['dataset_name'] not in relevant_datasets:
                    truncated_desc = truncate_description(pbs_row['dataset_description'])
                    dataset_text = (
                        f"**Dataset ID:** {pbs_row['dataset_id']}\n\n"
                        f"**Dataset Name:** {pbs_row['dataset_name']}\n\n"
                        f"**Description:** {truncated_desc}\n\n"
                        f"**🔑 Key Variable:** DRG_TYPE_CD (Diagnosis Related Groups)\n\n"
                        f"**📊 Relevance:** Essential for medical condition analysis"
                    )
                    st.markdown(dataset_text)
                    
                    relevant_datasets.append(pbs_row['dataset_name'])
            else:
                # If PBS dataset not found, show general recommendation
                st.warning("**PBS Dataset Recommended:** Use PBS data with DRG_TYPE_CD variable for comprehensive medical condition analysis including diagnosis-related groups.")
            
            # Always show standard recommendation
            recommendation = medical_info.get('recommendation', 
                'PBS dataset contains DRG_TYPE_CD variable which captures diagnosis-related groups essential for analyzing medical conditions and related healthcare outcomes.')
            st.caption(f"💡 **Why PBS?** {recommendation}")
            
            st.markdown("---")
        
        return relevant_datasets
    
    def display_geographic_analysis_suggestion(self, geographic_info, search_engine=None, relevant_datasets=None, user_input=None, query_analyzer=None):
        """Display geographic analysis considerations and recommendations."""
        if not geographic_info.get('is_geographic', False):
            return
        
        st.success("🗺️ **Geographic Analysis Detected**")
        st.caption(f"*Confidence: {geographic_info.get('confidence', 0):.0%}*")
        
        if geographic_info.get('geographic_keywords'):
            keywords_text = ", ".join(geographic_info['geographic_keywords'])
            st.caption(f"*Geographic terms identified: {keywords_text}*")
        
        if geographic_info.get('analysis_type'):
            st.caption(f"*Analysis type: {geographic_info['analysis_type'].replace('_', ' ').title()}*")
        
        # Display geographic considerations
        st.markdown("---")
        st.markdown("## 🗺️ **Geographic Analysis Considerations**")
        st.info("For geographic analysis, consider the following important factors:")
        
        # Geographic recommendations
        recommendations = [
            "🔍 **Location Variables**: Ensure datasets contain geographic identifiers (state, region, postcode)",
            "📊 **Spatial Comparisons**: Look for variables that allow comparison across geographic areas",
            "🏘️ **Geographic Coverage**: Verify datasets cover the geographic scope of your analysis",
            "📍 **Administrative Boundaries**: Consider alignment with relevant administrative divisions",
            "🗃️ **Geographic Linkage**: Check if datasets can be linked by geographic identifiers"
        ]
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        # Show specific recommendation if available
        if geographic_info.get('recommendation'):
            st.caption(f"💡 **Geographic Analysis Note:** {geographic_info['recommendation']}")
        
        # Search for geographic variables using OpenAI suggestions
        if search_engine and relevant_datasets and user_input and query_analyzer:
            self._display_geographic_dimension_with_ai(
                search_engine, relevant_datasets, user_input, geographic_info, query_analyzer
            )
        
        st.markdown("---")
    
    def _display_geographic_dimension_with_ai(self, search_engine, relevant_datasets, user_input, geographic_info, query_analyzer):
        """Display geographic variables using OpenAI suggestions."""
        st.markdown("## 🗺️ **Geographic Dimension**")
        st.caption("*AI-selected geographic variables from the actual datasets for your analysis*")
        
        # Get AI suggestions for geographic variables
        with st.spinner("🤖 Analyzing actual variables in your datasets..."):
            try:
                ai_suggestions = query_analyzer.suggest_geographic_variables(
                    user_input, geographic_info, relevant_datasets
                )
            except Exception as e:
                st.error(f"Error getting AI suggestions: {e}")
                ai_suggestions = query_analyzer._default_geographic_variables_response()
        
        # Display AI analysis rationale
        if ai_suggestions.get('analysis_rationale'):
            st.info(f"**Analysis Rationale:** {ai_suggestions['analysis_rationale']}")
        
        # Display dataset sources if available
        if ai_suggestions.get('dataset_sources'):
            st.caption(f"**Source Datasets:** {', '.join(ai_suggestions['dataset_sources'])}")
        
        # Display selected variables
        selected_variables = ai_suggestions.get('selected_variables', [])
        variable_descriptions = ai_suggestions.get('variable_descriptions', [])
        
        if selected_variables:
            st.success(f"Found {len(selected_variables)} geographic variables specifically selected for your analysis:")
            
            # Ensure descriptions list is same length as variables list
            if len(variable_descriptions) < len(selected_variables):
                variable_descriptions.extend(['Description not available'] * (len(selected_variables) - len(variable_descriptions)))
            
            for i, (var_name, var_desc) in enumerate(zip(selected_variables, variable_descriptions), 1):
                # Try to find additional details from the search engine
                try:
                    # Search for exact variable name to get dataset and score info
                    search_results = search_engine.search_variables(var_name, "variable", boost_datasets=relevant_datasets)
                    
                    # Find exact match or best match
                    best_match = None
                    for result in search_results:
                        if result['row'].get('variable_name', '').lower() == var_name.lower():
                            best_match = result
                            break
                    
                    if not best_match and search_results:
                        best_match = search_results[0]  # Use best scoring match
                    
                    if best_match:
                        row = best_match['row']
                        score = best_match['score']
                        dataset_name = row.get('dataset', 'Unknown')
                        dataset_id = row.get('dataset_id', 'Unknown')
                        boost_indicator = " ⭐" if best_match.get('boosted', False) else ""
                        
                        var_text = (
                            f"**{i}.** **Variable:** {var_name} | "
                            f"**Dataset:** {dataset_id} - {dataset_name}{boost_indicator} | "
                            f"**Description:** {var_desc} | "
                            f"**Match Score:** {score:.3f}"
                        )
                    else:
                        var_text = (
                            f"**{i}.** **Variable:** {var_name} | "
                            f"**Description:** {var_desc} | "
                            f"**Status:** AI-recommended from dataset variables"
                        )
                    
                except Exception:
                    var_text = (
                        f"**{i}.** **Variable:** {var_name} | "
                        f"**Description:** {var_desc} | "
                        f"**Status:** AI-recommended from dataset variables"
                    )
                
                container = st.empty()
                self.ui.stream_text(container, var_text)
        else:
            st.warning("No specific geographic variables could be identified from the selected datasets.")
            st.caption("*The AI analysis did not find suitable geographic variables in the available data*")
        
        # Show debug info for AI suggestions
        with st.expander("Geographic Variables AI Analysis (Debug)"):
            st.json({k: v for k, v in ai_suggestions.items() if k != 'raw_response'})
            if ai_suggestions.get('raw_response'):
                st.text("Raw AI Response:")
                st.write(ai_suggestions['raw_response'])