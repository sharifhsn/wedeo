use wedeo_core::media_type::MediaType;
use wedeo_core::rational::Rational;

use crate::Filter;

/// Configuration for a link between two filter nodes.
#[derive(Debug, Clone)]
pub struct FilterLinkConfig {
    pub media_type: MediaType,
    pub time_base: Rational,
}

/// A link between two filter nodes.
#[derive(Debug)]
pub struct FilterLink {
    pub src_node: usize,
    pub src_pad: usize,
    pub dst_node: usize,
    pub dst_pad: usize,
    pub config: FilterLinkConfig,
}

/// A node in the filter graph.
pub struct FilterNode {
    pub name: String,
    pub filter: Box<dyn Filter>,
    pub input_links: Vec<usize>,
    pub output_links: Vec<usize>,
}

/// The filter graph — manages filter nodes and links.
/// Stub implementation for now.
pub struct FilterGraph {
    nodes: Vec<FilterNode>,
    links: Vec<FilterLink>,
}

impl FilterGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            links: Vec::new(),
        }
    }

    pub fn add_node(&mut self, name: String, filter: Box<dyn Filter>) -> usize {
        let index = self.nodes.len();
        self.nodes.push(FilterNode {
            name,
            filter,
            input_links: Vec::new(),
            output_links: Vec::new(),
        });
        index
    }

    pub fn add_link(
        &mut self,
        src_node: usize,
        src_pad: usize,
        dst_node: usize,
        dst_pad: usize,
        config: FilterLinkConfig,
    ) -> usize {
        let index = self.links.len();
        self.links.push(FilterLink {
            src_node,
            src_pad,
            dst_node,
            dst_pad,
            config,
        });
        self.nodes[src_node].output_links.push(index);
        self.nodes[dst_node].input_links.push(index);
        index
    }

    pub fn nb_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn nb_links(&self) -> usize {
        self.links.len()
    }
}

impl Default for FilterGraph {
    fn default() -> Self {
        Self::new()
    }
}
