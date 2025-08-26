from config.topics import HIGHER_ED_KEYWORDS, DISABILITY_TOPICS

class SearchFilters:
    @staticmethod
    def is_higher_education_variable(description):
        """Check if variable is related to higher education."""
        return any(keyword in description.lower() for keyword in HIGHER_ED_KEYWORDS)
    
    @staticmethod
    def is_disability_topic(topic):
        """Check if topic is related to disability."""
        return topic.lower() in DISABILITY_TOPICS
    
    @staticmethod
    def is_alcd_variable(description):
        """Check if variable is likely from ALCD (using demographic as proxy)."""
        return "demography" in description.lower()
    
    @staticmethod
    def is_employment_variable(description):
        """Check if variable is employment-related."""
        return "(employment)" in description.lower()
    
    @staticmethod
    def is_demographic_variable(description):
        """Check if variable is demographic."""
        return "(demography)" in description.lower()
    
    @staticmethod
    def is_highered_variable(description):
        """Check if variable is explicitly marked as higher education."""
        return "(highered)" in description.lower()
    
    @staticmethod
    def get_appropriate_index(json_desc, topic):
        """Determine the appropriate search index based on variable description and topic."""
        filters = SearchFilters()
        
        if filters.is_highered_variable(json_desc):
            return 'he', " ", []
        elif filters.is_employment_variable(json_desc):
            additional_searches = [
                {"index": 'ato', "query_suffix": " income", "note": " (ATO income)"},
                {"index": 'domino', "query_suffix": " jobseeker", "note": " (DOMINO jobseeker)"}
            ]
            return 'non_acld', " ", additional_searches
        elif filters.is_demographic_variable(json_desc):
            return 'core', " (CORE dataset only)", []
        elif filters.is_alcd_variable(json_desc):
            return 'census', " (Census dataset only)", []
        elif filters.is_disability_topic(topic):
            return 'variables', "", []
        else:
            return 'non_ndis', "", []