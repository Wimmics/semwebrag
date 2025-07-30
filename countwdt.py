#count ce occurences of wdt: in file

import re
def count_wdt_occurrences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Count occurrences of 'wdt:' using regex
    wdt_occurrences = re.findall(r'\bwdt:\w+', content)
    
    return len(wdt_occurrences)
if __name__ == "__main__":
    file_path = 'medical/outputLinkerLinked.ttl'  # Replace with your file path
    count = count_wdt_occurrences(file_path)
    print(f'The file {file_path} contains {count} occurrences of "wdt:".')