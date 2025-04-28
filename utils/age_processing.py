
def get_sorted_ages(dataset, age_type):
    if dataset == 'lfm' or dataset == 'bx':
        if age_type == 'finegrained_age':
            ages_sort = ['12', '13', '14', '15', '16', '17', '18', '19-20', '21-22', '23-24', '25-29', '30-34', '35-44', '45-54', '55-65'] # Age group can be defined as a range (seperated by '_') or a single age
        elif age_type == 'binary_age':
            ages_sort = ['12-17', '18-65']
        elif age_type == 'finegrained_child_ages':
            ages_sort = ['12', '13', '14', '15', '16', '17', '18-65'] 
        elif age_type == 'all_ages':
            ages_sort = [str(i) for i in range(12, 66)]  # All ages from 12 to 65
    elif dataset == 'ml':
        if age_type == 'finegrained_age' or age_type == 'all_ages':
            ages_sort = ['Under 18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+']
        elif age_type == 'binary_age':
            ages_sort = ['Under 18', '18+']
        elif age_type == 'finegrained_child_ages':
            ages_sort = ['Under 18', '18-65']
    return ages_sort

# Define the age grouping function
def age_group(age, dataset, age_type):
    ages_sort = get_sorted_ages(dataset, age_type)
        
    if dataset == 'lfm' or dataset == 'bx':
        min_age = int(ages_sort[0].split('-')[0]) if '-' in ages_sort[0] else int(ages_sort[0])
        if age < min_age:
            return None  # Exclude ages below the minimum age in ages_sort
        else:
            for age_range in ages_sort:
                if '-' in age_range:
                    start_age, end_age = map(int, age_range.split('-'))
                    if start_age <= age <= end_age:
                        return age_range
                else: 
                    if age == int(age_range):
                        return age_range
            return None
        
    elif dataset == 'ml':
        if age == 1:
            return "Under 18"
        
        for age_group in ages_sort:
            if '-' in age_group:
                start_age, end_age = map(int, age_group.split('-'))
            elif 'Under' in age_group:
                start_age = 0
                end_age = int(age_group.split(' ')[1]) - 1
            elif '+' in age_group:
                start_age = int(age_group.split('+')[0])
                end_age = 100
            else: 
                return None
            
            if start_age <= age <= end_age:
                return age_group
        return None
        