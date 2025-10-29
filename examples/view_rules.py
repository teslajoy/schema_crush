"""view mapping rules interactively."""

from schema_crush.knowledge.mapping_rules import RuleDatabase, load_htan_rules, load_gdc_rules


def print_rule(rule):
    """print a mapping rule."""
    print(f"\n{rule.source_node} -> {rule.target_resource}")
    if rule.subclass_of:
        print(f"  subclass: {rule.subclass_of}")
    if rule.references:
        for ref in rule.references:
            print(f"  ref: {ref.resource}.{ref.field}")
    if rule.field_mappings:
        for fm in rule.field_mappings:
            print(f"  field: {fm.field} ({fm.type})")
            if fm.code:
                print(f"    code: {fm.code}")


# load rules
db = RuleDatabase()
db.add_rules(load_htan_rules())
db.add_rules(load_gdc_rules())

print(f"loaded {db.count()} rules\n")

# show examples
print("="*60)
print("observation rules (first 5):")
print("="*60)
for rule in db.find_by_target("observation")[:5]:
    print_rule(rule)

print("\n" + "="*60)
print("filtered: observation with specimen reference:")
print("="*60)
for rule in db.filter(target_resource="observation", has_reference_to="specimen")[:3]:
    print_rule(rule)