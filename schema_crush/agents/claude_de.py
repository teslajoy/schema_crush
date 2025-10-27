"""claude data engineer agent using anthropic api with markdown definition."""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from anthropic import Anthropic
import os


class ClaudeDataEngineer:
    """
    data engineer agent powered by anthropic claude.

    loads agent definition from markdown file and uses claude api
    for schema analysis and data profiling tasks.
    """

    def __init__(self, agent_path: Optional[str] = None, api_key: Optional[str] = None):
        """
        initialize claude data engineer agent.

        args:
            agent_path: path to agent definition markdown file
            api_key: anthropic api key (defaults to ANTHROPIC_API_KEY env var)
        """
        if agent_path is None:
            agent_path = Path(__file__).parent / "data_engineer.md"
        else:
            agent_path = Path(agent_path)

        self.agent_path = agent_path
        self.config = self._load_config()

        # initialize anthropic client
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        self.client = Anthropic(api_key=api_key)
        self.model = self._get_model_name()

    def _load_config(self) -> Dict[str, Any]:
        """load agent configuration from markdown file."""
        content = self.agent_path.read_text()

        # extract yaml frontmatter between --- markers
        parts = content.split('---')
        if len(parts) < 3:
            raise ValueError(f"invalid markdown format in {self.agent_path}")

        yaml_content = parts[1]
        config = yaml.safe_load(yaml_content)

        # get system prompt (everything after second ---)
        config['system_prompt'] = '---'.join(parts[2:]).strip()

        return config

    def _get_model_name(self) -> str:
        """map model shorthand to full model name."""
        model_map = {
            'sonnet': 'claude-3-5-sonnet-20241022',
            'opus': 'claude-3-opus-20240229',
            'haiku': 'claude-3-haiku-20240307'
        }

        model_key = self.config.get('model', 'sonnet')
        return model_map.get(model_key, model_key)

    def analyze_schema(
        self,
        source_schema: Dict[str, Any],
        target_schema: Dict[str, Any],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        analyze schema compatibility and mapping suggestions.

        args:
            source_schema: source data schema
            target_schema: target schema (e.g., fhir)
            context: optional context for the analysis

        returns:
            analysis results with compatibility scores and suggestions
        """
        prompt = self._build_schema_analysis_prompt(
            source_schema, target_schema, context
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=self.config['system_prompt'],
            messages=[{"role": "user", "content": prompt}]
        )

        return self._parse_response(response)

    def profile_field(
        self,
        field_name: str,
        field_data: Dict[str, Any],
        target_field: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        profile a field and assess mapping compatibility.

        args:
            field_name: source field name
            field_data: field metadata and samples
            target_field: optional target field to assess compatibility

        returns:
            field analysis with type, pattern, and compatibility score
        """
        prompt = self._build_field_profile_prompt(
            field_name, field_data, target_field
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=self.config['system_prompt'],
            messages=[{"role": "user", "content": prompt}]
        )

        return self._parse_response(response)

    def suggest_mappings(
        self,
        source_data: Dict[str, Any],
        target_type: str = "fhir"
    ) -> Dict[str, Any]:
        """
        suggest field mappings from source to target schema.

        args:
            source_data: source data with fields and samples
            target_type: target schema type (default: fhir)

        returns:
            mapping suggestions with confidence scores
        """
        prompt = self._build_mapping_prompt(source_data, target_type)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=self.config['system_prompt'],
            messages=[{"role": "user", "content": prompt}]
        )

        return self._parse_response(response)

    def _build_schema_analysis_prompt(
        self,
        source_schema: Dict[str, Any],
        target_schema: Dict[str, Any],
        context: Optional[str]
    ) -> str:
        """build prompt for schema analysis."""
        prompt = f"""analyze schema compatibility and suggest mappings.

source schema:
{json.dumps(source_schema, indent=2)}

target schema:
{json.dumps(target_schema, indent=2)}
"""

        if context:
            prompt += f"\ncontext: {context}"

        prompt += """

provide analysis in json format:
{
  "compatibility_score": 0.0-1.0,
  "entity_mappings": {
    "source_entity": "target_entity"
  },
  "field_mappings": {
    "source.field": "target.field"
  },
  "issues": ["list of compatibility issues"],
  "recommendations": ["list of recommendations"]
}
"""
        return prompt

    def _build_field_profile_prompt(
        self,
        field_name: str,
        field_data: Dict[str, Any],
        target_field: Optional[str]
    ) -> str:
        """build prompt for field profiling."""
        prompt = f"""profile field and assess mapping compatibility.

field name: {field_name}
field data:
{json.dumps(field_data, indent=2)}
"""

        if target_field:
            prompt += f"\ntarget field: {target_field}"

        prompt += """

provide analysis in json format:
{
  "field_analysis": {
    "type": "detected type",
    "is_identifier": true/false,
    "pattern": "detected pattern",
    "statistics": {}
  },
  "compatibility_score": 0.0-1.0,
  "reasoning": "explanation of score",
  "transform_needed": "none|simple|complex"
}
"""
        return prompt

    def _build_mapping_prompt(
        self,
        source_data: Dict[str, Any],
        target_type: str
    ) -> str:
        """build prompt for mapping suggestions."""
        return f"""suggest field mappings from source to {target_type}.

source data:
{json.dumps(source_data, indent=2)}

provide mapping suggestions in json format:
{{
  "mappings": {{
    "source.field": {{
      "target": "target.field",
      "confidence": 0.0-1.0,
      "reasoning": "explanation"
    }}
  }},
  "overall_confidence": 0.0-1.0
}}
"""

    def _parse_response(self, response) -> Dict[str, Any]:
        """parse claude response to json."""
        try:
            # get text content from response
            content = response.content[0].text

            # try to extract json from markdown code blocks
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()

            return json.loads(content)
        except (json.JSONDecodeError, IndexError, AttributeError) as e:
            # fallback: return raw response
            return {
                "error": f"failed to parse response: {e}",
                "raw_response": response.content[0].text if response.content else ""
            }