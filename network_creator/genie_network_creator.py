import pandas as pd
import datetime
import pysmile
import pysmile_license

smile = pysmile


def load_data(structure_path, cpt_dir):
    def list_parents(list_str):
        if not isinstance(list_str, str):
            list_str = str(list_str)

        return [int(p) for p in list_str.strip().split(";")] if list_str else []

    df = pd.read_csv(structure_path)
    df["parents_id"] = df["parents_id"].fillna("")
    df["parents_id"] = df["parents_id"].apply(list_parents)
    df_indexed = df.set_index("node_id")

    cpt_data = {}
    for idx, row in df_indexed.iterrows():
        node_id = idx
        cpt_file = f"{cpt_dir}/{node_id}_CPT.csv"
        cpt_data[node_id] = pd.read_csv(cpt_file, header=None, index_col=False)
    return df_indexed, cpt_data


def create_pysmile_network(df_indexed, cpt_data):
    net = smile.Network()
    node_id_to_handle = {}

    for idx, row in df_indexed.iterrows():
        handle = net.add_node(smile.NodeType.CPT, row["node_name"])
        net.set_node_id(handle, f"ID{str(idx)}")
        node_id_to_handle[idx] = handle

        states = [s for s in cpt_data[idx][0]]
        state_count = net.get_outcome_count(handle)

        for i in range(state_count):
            net.set_outcome_id(handle, f"State{str(i)}", states[i])

        while state_count < len(states):
            net.add_outcome(handle, states[state_count])
            state_count += 1

    for idx, row in df_indexed.iterrows():
        child_handle = node_id_to_handle[idx]
        if row["parents_id"] and row["parents_id"] != []:
            parents_ids = row["parents_id"]
            for p_id in parents_ids:
                parent_handle = node_id_to_handle[p_id]
                net.add_arc(parent_handle, child_handle)

    return net, node_id_to_handle


def set_node_definitions(net, df_indexed, cpt_data, node_id_to_handle):
    for idx, row in df_indexed.iterrows():
        node_handle = node_id_to_handle[idx]
        cpt_df = cpt_data[idx]

        if row["parents_id"] and row["parents_id"] != []:
            cpt_matrix = cpt_df.iloc[:, 1:]
            probabilities = cpt_matrix.transpose().values.flatten().tolist()
        else:
            probabilities = cpt_df[1].to_list()
        net.set_node_definition(node_handle, probabilities)

    # net.update_beliefs()

def calculate_node_positions(df_indexed):
    node_layers = {}
    assigned_nodes = set()

    current_layer = 0
    node_layers[current_layer] = []
    for idx, row in df_indexed.iterrows():
        if not row["parents_id"]:
            node_layers[current_layer].append(idx)
            assigned_nodes.add(idx)

    while len(assigned_nodes) < len(df_indexed):
        current_layer += 1
        node_layers[current_layer] = []
        nodes_added_in_current_layer = False
        for idx, row in df_indexed.iterrows():
            if idx not in assigned_nodes:
                parents = row["parents_id"]
                all_parents_assigned = True
                if parents:
                    for p_id in parents:
                        if p_id not in assigned_nodes:
                            all_parents_assigned = False
                            break

                if all_parents_assigned:
                    is_ready_for_this_layer = True
                    if parents and current_layer > 0:
                        parents_in_prev_layer = False
                        for p_id in parents:
                            if p_id in node_layers.get(current_layer - 1, []):
                                parents_in_prev_layer = True
                                break
                        if not parents_in_prev_layer and len(parents) > 0:
                             is_ready_for_this_layer = False

                    if is_ready_for_this_layer:
                        node_layers[current_layer].append(idx)
                        assigned_nodes.add(idx)
                        nodes_added_in_current_layer = True
        if not nodes_added_in_current_layer:
            del node_layers[current_layer]
            break

    return node_layers

def set_node_positions_in_network(net, node_layers, node_id_to_handle,
                                   node_width=120, node_height=60,
                                   horizontal_spacing=50, vertical_spacing=80,
                                   start_x=50, start_y=50):
    current_y = start_y
    for layer_num in sorted(node_layers.keys()):
        layer_nodes = node_layers[layer_num]
        current_x = start_x
        for node_idx in layer_nodes:
            handle = node_id_to_handle[node_idx]
            net.set_node_position(handle, current_x, current_y, node_width, node_height)
            current_x += node_width + horizontal_spacing
        current_y += node_height + vertical_spacing
    # print(f"Node positions set. Layers: {node_layers}")


def create_network(structure_path, cpt_dir, output_path="xdsl_files/out.xdsl", generate_input_file=False, log_flag=True):
    if log_flag:
        start_time = datetime.datetime.now()
        print("=" * 70)
        print(" BAYESIAN NETWORK CREATION PROCESS ".center(70, '='))
        print("=" * 70)
        print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 70)

        df_indexed = None
        cpt_data = None
        net = None
        node_id_to_handle = None

        try:
            # 1. Load Data
            print("\n[STEP 1/5]: Loading Network Data...")
            print("----------------------------------------------------------------------")
            df_indexed, cpt_data = load_data(structure_path, cpt_dir)
            print(f"SUCCESS: Data loaded from '{structure_path}' and '{cpt_dir}'.")
            print(f"         Found {len(df_indexed)} nodes and their CPTs.")

            # 2. Create PySmile Network Structure
            print("\n[STEP 2/5]: Creating PySmile Network Structure (Nodes and Arcs)...")
            print("----------------------------------------------------------------------")
            net, node_id_to_handle = create_pysmile_network(df_indexed, cpt_data)
            print("SUCCESS: PySmile network nodes and arcs created.")

            # 3. Set CPT Definitions
            print("\n[STEP 3/5]: Setting Conditional Probability Table (CPT) Definitions...")
            print("----------------------------------------------------------------------")
            set_node_definitions(net, df_indexed, cpt_data, node_id_to_handle)
            print("SUCCESS: Node definitions (CPTs) set and beliefs updated.")

            if generate_input_file:
                # 4. Calculate and Set Node Positions
                print("\n[STEP 4/5]: Calculating and Setting Node Positions for Visualization...")
                print("----------------------------------------------------------------------")
                node_layers = calculate_node_positions(df_indexed)
                set_node_positions_in_network(net, node_layers, node_id_to_handle)
                print(f"SUCCESS: Node positions set. Layout layers: {len(node_layers)}")

                # 5. Save the Network
                print("\n[STEP 5/5]: Saving the Network to XDSL File...")
                print("----------------------------------------------------------------------")
                net.write_file(output_path)
                print(f"SUCCESS: Network saved to '{output_path}'.")
            else:
                print("\n[INFO]: Skipping network position calculation and saving as 'generate_input_file' is False.")

        except FileNotFoundError as e:
            print(f"ERROR: Required file not found: {e}. Please ensure 'data/structure.csv' and CPT files exist.")
            return None
        except Exception as e:
            print(f"AN UNEXPECTED ERROR OCCURRED: {e}")
            return None

        finally:
            end_time = datetime.datetime.now()
            duration = end_time - start_time
            print("\n" + "=" * 70)
            print(" PROCESS SUMMARY ".center(70, '='))
            print("=" * 70)
            print(f"Process Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total Duration: {duration}")
            if df_indexed is not None:
                print(f"Number of Nodes Processed: {len(df_indexed)}")
            if generate_input_file:
                print(f"Output File: {output_path}")
            else:
                print("No output file generated (generate_input_file was False).")
            print("=" * 70)

            return net
    else:
        print("=" * 70)
        # 1. Load data
        df_indexed, cpt_data = load_data(structure_path, cpt_dir)
        print("Data loaded successfully.")

        # 2. Create PySmile network (add nodes and arcs)
        net, node_id_to_handle = create_pysmile_network(df_indexed, cpt_data)
        print("PySmile network structure created.")

        # 3. Set CPT definitions
        set_node_definitions(net, df_indexed, cpt_data, node_id_to_handle)
        print("Node definitions (CPTs) set.")

        # 4. Calculate and set node positions for visualization
        if log_flag:
            node_layers = calculate_node_positions(df_indexed)
            set_node_positions_in_network(net, node_layers, node_id_to_handle)

            # 5. Save the network
            net.write_file(output_xdsl_file)
            print(f"Network saved to {output_xdsl_file}")
        print("=" * 70)
        return net, node_id_to_handle

if __name__ == "__main__":
    structure_file = "../../data/out_structure.csv"
    cpt_directory = "../../data/CPTs/out_structure"
    output_xdsl_file = "xdsl_files/test.xdsl"
    create_network(structure_file, cpt_directory, output_xdsl_file, generate_input_file=True, log_flag=True)
