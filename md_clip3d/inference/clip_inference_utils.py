from __future__ import print_function

def normalize(values):
   max_value, min_value = max(values), min(values)
   new_values = [(value - min_value) / (max_value - min_value) for value in values]
   return new_values 
    
def remove_zero_prob_location(locations, probs, related_lists=None):
   target_indices = [idx for idx, prob in enumerate(probs) if prob != 0]
   probs = [probs[idx] for idx in target_indices]
   locations = [locations[idx] for idx in target_indices]
   if related_lists:
       new_lists = []
       for related_list in related_lists:
           new_lists.append([related_list[idx] for idx in target_indices])
       related_lists = new_lists 
   return locations, probs, related_lists

def map_fine_to_coarse(fine_label, coarse_labels, location_dict):
    fine_to_coarse_label = []
    all_fine_to_coarse_label = location_dict[fine_label]
    for coarse_label in coarse_labels:
        if coarse_label in all_fine_to_coarse_label:
            fine_to_coarse_label.append(coarse_label)
    if not len(fine_to_coarse_label):
        fine_to_coarse_label = all_fine_to_coarse_label[0]
    elif len(fine_to_coarse_label) > 1:
        for coarse_label in coarse_labels:
            if coarse_label in fine_to_coarse_label:
                fine_to_coarse_label = coarse_label
                break
    if isinstance(fine_to_coarse_label, list):
        fine_to_coarse_label = fine_to_coarse_label[0]
    return fine_to_coarse_label

def query_location(coarse, fine, location_dict):
   location = None
   for body in location_dict:
        if coarse in location_dict[body]:
            if fine in location_dict[body][coarse]:
                location = location_dict[body][coarse][fine]
   return location

if __name__ == '__main__':
    pass