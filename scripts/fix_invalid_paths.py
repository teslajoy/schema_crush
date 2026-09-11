#!/usr/bin/env python3
"""Fix invalid FHIR paths in flat_mappings.db.

Handles:
1. Case mismatches (Identifier -> identifier)
2. Invalid nested fields (Observation.DocumentReference.* -> proper paths)
3. Unknown resources (Extension.* -> proper paths)
4. Invalid field names (bodySite -> collection.bodySite)
5. FHIR R5 updates (medicationCodeableConcept -> medication)
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "schema_crush" / "data" / "db" / "flat_mappings.db"

# Define path corrections
PATH_FIXES = {
    # Case mismatches
    "DocumentReference.Identifier": "DocumentReference.identifier",
    "DocumentReference.Identifier.file_name": "DocumentReference.identifier.value",

    # Unknown resources -> proper paths
    "Extension.valueDateTime": "Observation.effectiveDateTime",
    "Extension.valueString": "Observation.valueString",
    "Extension.valueUnsignedInt": "Observation.valueInteger",
    "Identifier": "Patient.identifier",
    "Identifier.value": "Patient.identifier.value",
    "ResearchStudyProgressStatus.actual": "ResearchStudy.progressStatus.actual",
    "ResearchStudyRecruitment.actualNumber": "ResearchStudy.recruitment.actualNumber",

    # MedicationAdministration R5 fixes
    "MedicationAdministration.medicationCodeableConcept.coding": "MedicationAdministration.medication.concept.coding",
    "MedicationAdministration.occurrenceTiming": "MedicationAdministration.occurenceTiming",  # fhir.resources typo
    "MedicationAdministration.treatment_type": "MedicationAdministration.category.coding",
    "MedicationAdministration.valueString": "MedicationAdministration.note.text",

    # DocumentReference fixes
    "DocumentReference.component": "DocumentReference.content",

    # Specimen fixes
    "Specimen.bodySite": "Specimen.collection.bodySite",
    "Specimen.valueString": "Specimen.note.text",
    "Specimen.aliquot.center_id": "Specimen.identifier.value",
    "Specimen.aliquot.center_name": "Specimen.identifier.assigner.display",
    "Specimen.aliquot.namespace": "Specimen.identifier.system",
    "Specimen.aliquot.short_name": "Specimen.identifier.value",

    # Patient fixes
    "Patient.valueString": "Patient.extension.valueString",

    # ResearchStudy fixes
    "ResearchStudy.dbgap_accession_number": "ResearchStudy.identifier.value",

    # Observation.patient.* -> Observation fields (exposure data)
    "Observation.patient.pack_years_smoked": "Observation.valueQuantity",
    "Observation.patient.cigarettes_per_day": "Observation.valueQuantity",
    "Observation.patient.years_smoked": "Observation.valueQuantity",
    "Observation.patient.exposure_id": "Observation.identifier.value",
    "Observation.patient.alcohol_history": "Observation.valueCodeableConcept",
    "Observation.patient.alcohol_intensity": "Observation.valueCodeableConcept",

    # Observation.sample.* -> Specimen fields
    "Observation.sample.composition": "Specimen.type.coding",
    "Observation.sample.is_ffpe": "Specimen.processing.method.coding",
    "Observation.sample.preservation_method": "Specimen.processing.method.coding",
    "Observation.sample.updated_datetime": "Specimen.receivedTime",

    # Observation.portions.* -> Specimen fields
    "Observation.portions.is_ffpe": "Specimen.processing.method.coding",
    "Observation.portions.weight": "Specimen.collection.quantity",
    "Observation.portion.updated_datetime": "Specimen.receivedTime",

    # Observation.slides.* -> Specimen
    "Observation.slides.section_location": "Specimen.collection.bodySite.text",

    # Observation.survey.* -> Observation or Patient
    "Observation.survey.days_to_birth": "Patient.birthDate",
    "Observation.survey.days_to_death": "Patient.deceasedDateTime",
    "Observation.survey.days_to_diagnosis": "Condition.onsetDateTime",
    "Observation.survey.days_to_last_follow_up": "Observation.effectiveDateTime",
    "Observation.survey.days_to_last_known_disease_status": "Observation.effectiveDateTime",
    "Observation.survey.updated_datetime": "Observation.issued",

    # Observation.aliquot.* -> Specimen fields
    "Observation.aliquot.aliquot_quantity": "Specimen.collection.quantity",
    "Observation.aliquot.aliquot_volume": "Specimen.collection.quantity",
    "Observation.aliquot.analyte_type": "Specimen.type.coding",
    "Observation.aliquot.concentration": "Observation.valueQuantity",
    "Observation.aliquot.no_matched_normal_low_pass_wgs": "Observation.valueBoolean",
    "Observation.aliquot.no_matched_normal_targeted_sequencing": "Observation.valueBoolean",
    "Observation.aliquot.no_matched_normal_wgs": "Observation.valueBoolean",
    "Observation.aliquot.no_matched_normal_wxs": "Observation.valueBoolean",
    "Observation.aliquot.selected_normal_low_pass_wgs": "Observation.valueString",
    "Observation.aliquot.selected_normal_targeted_sequencing": "Observation.valueString",
    "Observation.aliquot.selected_normal_wgs": "Observation.valueString",
    "Observation.aliquot.selected_normal_wxs": "Observation.valueString",
    "Observation.aliquot.updated_datetime": "Observation.issued",

    # Observation.analyte.* -> Observation fields
    "Observation.analyte.a260_a280_ratio": "Observation.valueQuantity",
    "Observation.analyte.concentration": "Observation.valueQuantity",
    "Observation.analyte.experimental_protocol_type": "Observation.method.coding",
    "Observation.analyte.normal_tumor_genotype_snp_match": "Observation.valueString",
    "Observation.analyte.ribosomal_rna_28s_16s_ratio": "Observation.valueQuantity",
    "Observation.analyte.rna_integrity_number": "Observation.valueQuantity",
    "Observation.analyte.spectrophotometer_method": "Observation.method.coding",
    "Observation.analyte.updated_datetime": "Observation.issued",

    # Observation.DocumentReference.* -> DocumentReference fields (sequencing metadata)
    "Observation.DocumentReference.RIN": "Observation.valueQuantity",
    "Observation.DocumentReference.adapter_name": "DocumentReference.description",
    "Observation.DocumentReference.adapter_sequence": "DocumentReference.content.attachment.data",
    "Observation.DocumentReference.base_caller_name": "DocumentReference.description",
    "Observation.DocumentReference.base_caller_version": "DocumentReference.description",
    "Observation.DocumentReference.created_datetime": "DocumentReference.date",
    "Observation.DocumentReference.days_to_sequencing": "Observation.effectiveDateTime",
    "Observation.DocumentReference.experiment_name": "DocumentReference.identifier.value",
    "Observation.DocumentReference.flow_cell_barcode": "DocumentReference.identifier.value",
    "Observation.DocumentReference.fragment_maximum_length": "Observation.valueQuantity",
    "Observation.DocumentReference.fragment_mean_length": "Observation.valueQuantity",
    "Observation.DocumentReference.fragment_minimum_length": "Observation.valueQuantity",
    "Observation.DocumentReference.fragment_standard_deviation_length": "Observation.valueQuantity",
    "Observation.DocumentReference.includes_spike_ins": "Observation.valueBoolean",
    "Observation.DocumentReference.instrument_model": "DocumentReference.description",
    "Observation.DocumentReference.is_paired_end": "Observation.valueBoolean",
    "Observation.DocumentReference.lane_number": "DocumentReference.identifier.value",
    "Observation.DocumentReference.library_name": "DocumentReference.identifier.value",
    "Observation.DocumentReference.library_preparation_kit_catalog_number": "DocumentReference.identifier.value",
    "Observation.DocumentReference.library_preparation_kit_name": "DocumentReference.description",
    "Observation.DocumentReference.library_preparation_kit_vendor": "DocumentReference.author.display",
    "Observation.DocumentReference.library_preparation_kit_version": "DocumentReference.description",
    "Observation.DocumentReference.library_selection": "DocumentReference.category.coding",
    "Observation.DocumentReference.library_strand": "DocumentReference.description",
    "Observation.DocumentReference.library_strategy": "DocumentReference.category.coding",
    "Observation.DocumentReference.multiplex_barcode": "DocumentReference.identifier.value",
    "Observation.DocumentReference.number_expect_cells": "Observation.valueInteger",
    "Observation.DocumentReference.platform": "DocumentReference.description",
    "Observation.DocumentReference.read_group_id": "DocumentReference.identifier.value",
    "Observation.DocumentReference.read_group_name": "DocumentReference.identifier.value",
    "Observation.DocumentReference.read_length": "Observation.valueInteger",
    "Observation.DocumentReference.sequencing_center": "DocumentReference.custodian.display",
    "Observation.DocumentReference.sequencing_date": "DocumentReference.date",
    "Observation.DocumentReference.single_cell_library": "DocumentReference.category.coding",
    "Observation.DocumentReference.size_selection_range": "DocumentReference.description",
    "Observation.DocumentReference.spike_ins_concentration": "Observation.valueQuantity",
    "Observation.DocumentReference.spike_ins_fasta": "DocumentReference.content.attachment.url",
    "Observation.DocumentReference.state": "DocumentReference.status",
    "Observation.DocumentReference.submitter_id": "DocumentReference.identifier.value",
    "Observation.DocumentReference.target_capture_kit": "DocumentReference.description",
    "Observation.DocumentReference.target_capture_kit_catalog_number": "DocumentReference.identifier.value",
    "Observation.DocumentReference.target_capture_kit_name": "DocumentReference.description",
    "Observation.DocumentReference.target_capture_kit_target_region": "DocumentReference.description",
    "Observation.DocumentReference.target_capture_kit_vendor": "DocumentReference.author.display",
    "Observation.DocumentReference.target_capture_kit_version": "DocumentReference.description",
    "Observation.DocumentReference.to_trim_adapter_sequence": "Observation.valueBoolean",
    "Observation.DocumentReference.updated_datetime": "DocumentReference.date",
}

# Resource-only mappings that are valid for entity tier (leave as-is)
ENTITY_TIER_VALID = {
    "Condition", "Device", "DocumentReference", "Medication",
    "MedicationAdministration", "Observation", "Patient",
    "ResearchStudy", "ServiceRequest", "Specimen"
}


def fix_paths(db_path: Path = DB_PATH, dry_run: bool = True):
    """Apply path fixes to database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print(f"{'DRY RUN: ' if dry_run else ''}Fixing invalid paths in {db_path}")
    print("=" * 70)

    # Get current invalid paths
    cur.execute("SELECT DISTINCT destination FROM destinations")
    all_destinations = {row[0] for row in cur.fetchall()}

    fixed_count = 0
    skipped_entity = 0
    not_found = []

    for old_path, new_path in PATH_FIXES.items():
        if old_path in all_destinations:
            cur.execute(
                "SELECT COUNT(*) FROM destinations WHERE destination = ?",
                (old_path,)
            )
            count = cur.fetchone()[0]

            print(f"  {old_path}")
            print(f"    -> {new_path} ({count} records)")

            if not dry_run:
                cur.execute(
                    "UPDATE destinations SET destination = ? WHERE destination = ?",
                    (new_path, old_path)
                )
            fixed_count += count
        else:
            not_found.append(old_path)

    # Report entity-tier paths (valid, no fix needed)
    for entity in ENTITY_TIER_VALID:
        if entity in all_destinations:
            cur.execute(
                "SELECT COUNT(*) FROM destinations WHERE destination = ?",
                (entity,)
            )
            count = cur.fetchone()[0]
            print(f"  {entity} (entity-tier, valid) - {count} records")
            skipped_entity += count

    if not dry_run:
        conn.commit()

    conn.close()

    print("\n" + "=" * 70)
    print(f"SUMMARY:")
    print(f"  Fixed: {fixed_count} records")
    print(f"  Entity-tier (valid): {skipped_entity} records")
    print(f"  Fix rules not found in DB: {len(not_found)}")

    if dry_run:
        print("\nRun with --apply to apply changes")
    else:
        print("\nChanges applied successfully")

    return fixed_count


def verify_fixes(db_path: Path = DB_PATH):
    """Verify that fixes were applied correctly."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print("\nVERIFYING FIXES")
    print("=" * 70)

    # Check for any remaining invalid paths that we have fixes for
    remaining = []
    for old_path in PATH_FIXES.keys():
        cur.execute(
            "SELECT COUNT(*) FROM destinations WHERE destination = ?",
            (old_path,)
        )
        count = cur.fetchone()[0]
        if count > 0:
            remaining.append((old_path, count))

    if remaining:
        print("WARNING: These paths still exist (fixes not applied):")
        for path, count in remaining:
            print(f"  {path}: {count} records")
    else:
        print("All fixable paths have been corrected")

    conn.close()
    return len(remaining) == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fix invalid FHIR paths")
    parser.add_argument("--apply", action="store_true", help="Apply fixes (default: dry run)")
    parser.add_argument("--verify", action="store_true", help="Verify fixes were applied")
    parser.add_argument("--db", type=str, help="Path to database", default=str(DB_PATH))

    args = parser.parse_args()
    db_path = Path(args.db)

    if args.verify:
        verify_fixes(db_path)
    else:
        fix_paths(db_path, dry_run=not args.apply)