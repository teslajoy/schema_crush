#!/usr/bin/env python3
"""audit flat_mappings.db for issues.

reports:
- invalid FHIR paths (custom paths like Observation.code.nci_tumor_grade)
- duplicate sources with different destinations
- sources with no destinations
- breakdown by schema
"""

import sqlite3
import sys
from pathlib import Path
from collections import defaultdict

# add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "schema_crush" / "data" / "db" / "flat_mappings.db"


def get_fhir_resources():
    """get valid FHIR R5 resources from fhir.resources package."""
    try:
        # fhir.resources 8.0.0b4 = FHIR R5 (5.0.0)
        from fhir.resources import get_fhir_model_class

        # common FHIR R5 resources
        resources = {
            "Account", "ActivityDefinition", "ActorDefinition", "AdministrableProductDefinition",
            "AdverseEvent", "AllergyIntolerance", "Appointment", "AppointmentResponse",
            "ArtifactAssessment", "AuditEvent", "Basic", "Binary", "BiologicallyDerivedProduct",
            "BiologicallyDerivedProductDispense", "BodyStructure", "Bundle", "CapabilityStatement",
            "CarePlan", "CareTeam", "ChargeItem", "ChargeItemDefinition", "Citation", "Claim",
            "ClaimResponse", "ClinicalImpression", "ClinicalUseDefinition", "CodeSystem",
            "Communication", "CommunicationRequest", "CompartmentDefinition", "Composition",
            "ConceptMap", "Condition", "ConditionDefinition", "Consent", "Contract", "Coverage",
            "CoverageEligibilityRequest", "CoverageEligibilityResponse", "DetectedIssue", "Device",
            "DeviceAssociation", "DeviceDefinition", "DeviceDispense", "DeviceMetric", "DeviceRequest",
            "DeviceUsage", "DiagnosticReport", "DocumentReference", "Encounter", "EncounterHistory",
            "Endpoint", "EnrollmentRequest", "EnrollmentResponse", "EpisodeOfCare", "EventDefinition",
            "Evidence", "EvidenceReport", "EvidenceVariable", "ExampleScenario", "ExplanationOfBenefit",
            "FamilyMemberHistory", "Flag", "FormularyItem", "GenomicStudy", "Goal", "GraphDefinition",
            "Group", "GuidanceResponse", "HealthcareService", "ImagingSelection", "ImagingStudy",
            "Immunization", "ImmunizationEvaluation", "ImmunizationRecommendation", "ImplementationGuide",
            "Ingredient", "InsurancePlan", "InventoryItem", "InventoryReport", "Invoice", "Library",
            "Linkage", "List", "Location", "ManufacturedItemDefinition", "Measure", "MeasureReport",
            "Medication", "MedicationAdministration", "MedicationDispense", "MedicationKnowledge",
            "MedicationRequest", "MedicationStatement", "MedicinalProductDefinition", "MessageDefinition",
            "MessageHeader", "MolecularSequence", "NamingSystem", "NutritionIntake", "NutritionOrder",
            "NutritionProduct", "Observation", "ObservationDefinition", "OperationDefinition",
            "OperationOutcome", "Organization", "OrganizationAffiliation", "PackagedProductDefinition",
            "Parameters", "Patient", "PaymentNotice", "PaymentReconciliation", "Permission", "Person",
            "PlanDefinition", "Practitioner", "PractitionerRole", "Procedure", "Provenance",
            "Questionnaire", "QuestionnaireResponse", "RegulatedAuthorization", "RelatedPerson",
            "RequestOrchestration", "Requirements", "ResearchStudy", "ResearchSubject", "RiskAssessment",
            "Schedule", "SearchParameter", "ServiceRequest", "Slot", "Specimen", "SpecimenDefinition",
            "StructureDefinition", "StructureMap", "Subscription", "SubscriptionStatus",
            "SubscriptionTopic", "Substance", "SubstanceDefinition", "SubstanceNucleicAcid",
            "SubstancePolymer", "SubstanceProtein", "SubstanceReferenceInformation", "SubstanceSourceMaterial",
            "SupplyDelivery", "SupplyRequest", "Task", "TerminologyCapabilities", "TestPlan", "TestReport",
            "TestScript", "Transport", "ValueSet", "VerificationResult", "VisionPrescription",
        }
        return resources
    except ImportError:
        print("warning: fhir.resources not installed, using fallback list")
        return {
            "Patient", "Specimen", "Observation", "Condition", "Procedure",
            "DiagnosticReport", "DocumentReference", "ResearchStudy",
            "ResearchSubject", "MedicationAdministration", "Encounter",
            "Organization", "Practitioner", "Group", "Task", "BodyStructure",
        }


def get_fhir_fields(resource: str):
    """get valid fields for a FHIR R5 resource using fhir.resources."""
    try:
        from fhir.resources import get_fhir_model_class

        model_class = get_fhir_model_class(resource)
        if model_class is None:
            return set()

        # get field names from pydantic model
        fields = set(model_class.model_fields.keys())
        return fields
    except Exception as e:
        # fallback to schema explorer
        try:
            from schema_crush.tools.fhir_schema_tool import get_schema_explorer
            explorer = get_schema_explorer()
            return set(explorer.get_resource_fields(resource))
        except Exception:
            return set()


def validate_fhir_path(path: str, valid_resources: set) -> dict:
    """validate a FHIR path like 'Observation.code.coding'.

    returns dict with:
        valid: bool
        resource: str
        field: str
        issue: str (if invalid)
    """
    result = {"path": path, "valid": False, "resource": "", "field": "", "issue": ""}

    if not path or "." not in path:
        result["issue"] = "no dot separator"
        return result

    parts = path.split(".")
    resource = parts[0]

    # check resource
    if resource not in valid_resources:
        # case-insensitive check
        matched = [r for r in valid_resources if r.lower() == resource.lower()]
        if matched:
            result["issue"] = f"case mismatch: should be '{matched[0]}'"
            result["resource"] = matched[0]
        else:
            result["issue"] = f"unknown resource: {resource}"
        return result

    result["resource"] = resource

    # check first-level field
    if len(parts) > 1:
        field = parts[1]
        valid_fields = get_fhir_fields(resource)

        if valid_fields and field not in valid_fields:
            # check for custom/non-standard field
            if field.startswith("nci_") or field.startswith("gdc_") or field.startswith("htan_"):
                result["issue"] = f"custom field: {field} (not standard FHIR)"
            else:
                # case-insensitive check
                matched = [f for f in valid_fields if f.lower() == field.lower()]
                if matched:
                    result["issue"] = f"case mismatch: should be '{matched[0]}'"
                else:
                    result["issue"] = f"unknown field: {field}"
            return result

        result["field"] = field

    result["valid"] = True
    return result


def audit_db(db_path: Path = DB_PATH):
    """run full audit on flat_mappings.db."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    print("=" * 70)
    print("FLAT MAPPINGS DB AUDIT")
    print("=" * 70)

    # 1. basic stats
    print("\n1. BASIC STATS")
    print("-" * 40)

    cur.execute("SELECT COUNT(*) FROM sources")
    source_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM destinations")
    dest_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM content_values")
    cv_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM content_fhir_targets")
    cft_count = cur.fetchone()[0]

    print(f"  sources:             {source_count:,}")
    print(f"  destinations:        {dest_count:,}")
    print(f"  content_values:      {cv_count:,}")
    print(f"  content_fhir_targets:{cft_count:,}")

    # 2. breakdown by schema
    print("\n2. SOURCES BY SCHEMA")
    print("-" * 40)

    cur.execute("""
        SELECT source_schema, tier, COUNT(*) as cnt
        FROM sources
        GROUP BY source_schema, tier
        ORDER BY source_schema, tier
    """)

    schema_stats = defaultdict(dict)
    for row in cur.fetchall():
        schema_stats[row["source_schema"]][row["tier"]] = row["cnt"]

    for schema, tiers in sorted(schema_stats.items()):
        total = sum(tiers.values())
        print(f"  {schema}: {total} total")
        for tier, cnt in sorted(tiers.items()):
            print(f"    - {tier}: {cnt}")

    # 3. validate FHIR paths
    print("\n3. FHIR PATH VALIDATION")
    print("-" * 40)

    valid_resources = get_fhir_resources()
    print(f"  loaded {len(valid_resources)} FHIR resources")

    cur.execute("SELECT DISTINCT destination FROM destinations")
    destinations = [row["destination"] for row in cur.fetchall()]

    invalid_paths = []
    custom_paths = []
    valid_count = 0

    for dest in destinations:
        result = validate_fhir_path(dest, valid_resources)
        if result["valid"]:
            valid_count += 1
        elif "custom field" in result.get("issue", ""):
            custom_paths.append((dest, result["issue"]))
        else:
            invalid_paths.append((dest, result["issue"]))

    print(f"  valid paths:   {valid_count}")
    print(f"  custom paths:  {len(custom_paths)} (non-standard FHIR)")
    print(f"  invalid paths: {len(invalid_paths)}")

    if custom_paths:
        print("\n  CUSTOM PATHS (non-standard):")
        for path, issue in custom_paths[:20]:
            print(f"    {path}")
            print(f"      -> {issue}")
        if len(custom_paths) > 20:
            print(f"    ... and {len(custom_paths) - 20} more")

    if invalid_paths:
        print("\n  INVALID PATHS:")
        for path, issue in invalid_paths[:20]:
            print(f"    {path}")
            print(f"      -> {issue}")
        if len(invalid_paths) > 20:
            print(f"    ... and {len(invalid_paths) - 20} more")

    # 4. duplicate sources (same source, different destinations)
    print("\n4. SOURCES WITH MULTIPLE DESTINATIONS")
    print("-" * 40)

    cur.execute("""
        SELECT s.source, s.source_schema, COUNT(DISTINCT d.destination) as dest_count,
               GROUP_CONCAT(DISTINCT d.destination) as destinations
        FROM sources s
        JOIN destinations d ON s.id = d.source_id
        GROUP BY s.source, s.source_schema
        HAVING dest_count > 1
        ORDER BY dest_count DESC
        LIMIT 20
    """)

    multi_dest = cur.fetchall()
    print(f"  found {len(multi_dest)} sources with multiple destinations")

    for row in multi_dest[:10]:
        print(f"\n  {row['source']} ({row['source_schema']}): {row['dest_count']} destinations")
        for dest in row["destinations"].split(","):
            print(f"    -> {dest}")

    # 5. sources with no destinations
    print("\n5. ORPHAN SOURCES (no destinations)")
    print("-" * 40)

    cur.execute("""
        SELECT s.id, s.source, s.source_schema
        FROM sources s
        LEFT JOIN destinations d ON s.id = d.source_id
        WHERE d.id IS NULL
    """)

    orphans = cur.fetchall()
    print(f"  found {len(orphans)} sources with no destinations")

    for row in orphans[:10]:
        print(f"    {row['source']} ({row['source_schema']})")

    # 6. destination uniqueness
    print("\n6. MOST COMMON DESTINATIONS")
    print("-" * 40)

    cur.execute("""
        SELECT destination, COUNT(*) as cnt
        FROM destinations
        GROUP BY destination
        ORDER BY cnt DESC
        LIMIT 15
    """)

    for row in cur.fetchall():
        print(f"  {row['destination']}: {row['cnt']} sources")

    conn.close()

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    return {
        "source_count": source_count,
        "dest_count": dest_count,
        "valid_paths": valid_count,
        "custom_paths": len(custom_paths),
        "invalid_paths": len(invalid_paths),
        "multi_dest_sources": len(multi_dest),
        "orphan_sources": len(orphans),
    }


def export_issues(db_path: Path = DB_PATH, output_dir: Path = None):
    """export issues to CSV files for review."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "audit_output"
    output_dir.mkdir(exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    valid_resources = get_fhir_resources()

    # export all mappings with validation status
    print(f"\nexporting to {output_dir}/...")

    cur.execute("""
        SELECT s.id, s.source, s.source_schema, s.tier, s.source_context,
               d.destination, d.dest_system, d.dest_code
        FROM sources s
        JOIN destinations d ON s.id = d.source_id
        ORDER BY s.source_schema, s.source
    """)

    with open(output_dir / "all_mappings_validated.csv", "w") as f:
        f.write("source_id,source,schema,tier,context,destination,dest_system,dest_code,fhir_valid,issue\n")

        for row in cur.fetchall():
            result = validate_fhir_path(row["destination"], valid_resources)
            valid = "yes" if result["valid"] else "no"
            issue = result.get("issue", "").replace(",", ";")

            line = f'"{row["id"]}","{row["source"]}","{row["source_schema"]}","{row["tier"]}","{row["source_context"] or ""}","{row["destination"]}","{row["dest_system"] or ""}","{row["dest_code"] or ""}","{valid}","{issue}"\n'
            f.write(line)

    print(f"  saved: all_mappings_validated.csv")

    # export just invalid/custom paths
    cur.execute("SELECT DISTINCT destination FROM destinations")

    with open(output_dir / "invalid_paths.csv", "w") as f:
        f.write("destination,issue,suggested_fix\n")

        for row in cur.fetchall():
            result = validate_fhir_path(row["destination"], valid_resources)
            if not result["valid"]:
                issue = result.get("issue", "").replace(",", ";")
                f.write(f'"{row["destination"]}","{issue}",""\n')

    print(f"  saved: invalid_paths.csv")

    conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audit flat_mappings.db")
    parser.add_argument("--export", action="store_true", help="Export issues to CSV")
    parser.add_argument("--db", type=str, help="Path to database", default=str(DB_PATH))

    args = parser.parse_args()

    db_path = Path(args.db)

    audit_db(db_path)

    if args.export:
        export_issues(db_path)