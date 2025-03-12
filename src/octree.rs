use glam::{DVec3, UVec3};

#[derive(Clone, Copy)]
enum TreeNodeData {
    Data(Option<DVec3>),
    Children(usize),
}

#[derive(Clone)]
pub struct TreeNode {
    anchor: DVec3,
    width: DVec3,
    inner: TreeNodeData,
}

impl TreeNode {
    /// We can only directly create leaf nodes
    fn new(anchor: DVec3, width: DVec3, data: Option<DVec3>) -> Self {
        TreeNode {
            anchor,
            width,
            inner: TreeNodeData::Data(data),
        }
    }

    fn contains(&self, loc: DVec3) -> bool {
        let opposite = self.anchor + self.width;
        self.anchor.x <= loc.x
            && loc.x < opposite.x
            && self.anchor.y <= loc.y
            && loc.y < opposite.y
            && self.anchor.z <= loc.z
            && loc.z < opposite.z
    }

    fn is_leaf(&self) -> bool {
        match self.inner {
            TreeNodeData::Data(_) => true,
            TreeNodeData::Children(_) => false,
        }
    }

    pub fn data(&self) -> Option<DVec3> {
        match self.inner {
            TreeNodeData::Data(data) => data,
            TreeNodeData::Children(_) => None,
        }
    }

    pub fn anchor(&self) -> DVec3 {
        self.anchor
    }

    pub fn width(&self) -> DVec3 {
        self.width
    }
}

#[derive(Clone)]
pub struct Octree {
    nodes: Vec<TreeNode>,
    top_level_count: usize,
}

impl Octree {
    pub fn init_from_bg_resolution(
        box_size: DVec3,
        resolution: u32,
        box_center_shift: Option<DVec3>,
    ) -> Self {
        let box_center = box_center_shift.unwrap_or_default();
        let min_width = box_size.min_element();
        let resolution = UVec3::new(
            (box_size.x / min_width * resolution as f64).round() as u32,
            (box_size.x / min_width * resolution as f64).round() as u32,
            (box_size.x / min_width * resolution as f64).round() as u32,
        );
        let width_top = box_size / resolution.as_dvec3();
        let top_level_count = resolution.element_product() as usize;
        let anchor = -box_center;
        let mut nodes = Vec::with_capacity(top_level_count);
        for i in 0..resolution.x {
            let x = anchor.x + i as f64 * width_top.x;
            for j in 0..resolution.y {
                let y = anchor.y + j as f64 * width_top.y;
                for k in 0..resolution.z {
                    let anchor = DVec3::new(x, y, anchor.z + k as f64 * width_top.z);
                    nodes.push(TreeNode::new(anchor, width_top, None));
                }
            }
        }

        Self {
            nodes,
            top_level_count,
        }
    }

    pub fn init_from_bg_file(
        box_size: DVec3,
        file_name: &str,
        box_center_shift: Option<DVec3>,
    ) -> Result<Self, hdf5::Error> {
        let file = hdf5::File::open(file_name)?;
        let data = file.group("PartType0")?;
        let coordinates = data.dataset("Coordinates")?.read_raw::<f64>()?;
        let coordinates = coordinates
            .chunks(3)
            .map(DVec3::from_slice)
            .collect::<Vec<_>>();

        let mut nodes = Vec::with_capacity(coordinates.len());
        let anchor = -box_center_shift.unwrap_or_default();
        nodes.push(TreeNode::new(anchor, box_size, None));
        let mut tree = Self {
            nodes,
            top_level_count: 1,
        };
        for loc in coordinates {
            tree.insert(loc);
        }
        Ok(tree)
    }

    fn find_leaf_containing(&self, loc: DVec3) -> usize {
        let mut cur_node_idx = (0..self.top_level_count)
            .find(|idx| self.nodes[*idx].contains(loc))
            .expect("Location outside box size!");
        loop {
            cur_node_idx = match self.nodes[cur_node_idx].inner {
                TreeNodeData::Data(_) => return cur_node_idx,
                TreeNodeData::Children(children) => (children..children + 8)
                    .find(|idx| self.nodes[*idx].contains(loc))
                    .expect("At least one of the children must contain the location!"),
            };
        }
    }

    pub fn insert(&mut self, loc: DVec3) {
        let leaf_idx = self.find_leaf_containing(loc);
        self.insert_in_leaf(leaf_idx, loc);
    }

    fn insert_in_leaf(&mut self, leaf_idx: usize, loc: DVec3) {
        assert!(self.nodes[leaf_idx].is_leaf());
        let leaf_data = match self.nodes[leaf_idx].inner {
            TreeNodeData::Data(data) => data,
            TreeNodeData::Children(_) => unreachable!("A leaf node does not have children"),
        };
        match leaf_data {
            None => self.nodes[leaf_idx].inner = TreeNodeData::Data(Some(loc)),
            Some(leaf_data) => {
                // Convert to interior node
                let offset = self.nodes.len();
                self.nodes[leaf_idx].inner = TreeNodeData::Children(offset);

                // Create new children
                self.nodes.reserve(8);
                let anchor = self.nodes[leaf_idx].anchor;
                let cwidth = 0.5 * self.nodes[leaf_idx].width;
                for i in 0..2 {
                    let x = anchor.x + i as f64 * cwidth.x;
                    for j in 0..2 {
                        let y = anchor.y + j as f64 * cwidth.y;
                        for k in 0..2 {
                            let mut node = TreeNode::new(
                                DVec3::new(x, y, anchor.z + k as f64 * cwidth.z),
                                cwidth,
                                None,
                            );
                            if node.contains(leaf_data) {
                                node.inner = TreeNodeData::Data(Some(leaf_data))
                            }
                            self.nodes.push(node);
                        }
                    }
                }
                let next_leaf_idx = (offset..offset + 8)
                    .find(|idx| self.nodes[*idx].contains(loc))
                    .expect("One of the children must contain the new data");
                self.insert_in_leaf(next_leaf_idx, loc);
            }
        }
    }

    pub fn into_leaves(self) -> Vec<TreeNode> {
        self.nodes.into_iter().filter(|n| n.is_leaf()).collect()
    }
}
