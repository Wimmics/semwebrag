import wikidatautils

res = wikidatautils.get_entity_info("medical/outputLinkerLinked.ttl", "Aerosolization")
print("Entity Info:", res)