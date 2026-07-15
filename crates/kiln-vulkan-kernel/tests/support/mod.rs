use kiln_vulkan_kernel::VulkanDevice;
use std::sync::Arc;

fn qualification_required() -> bool {
    std::env::var("KILN_QUALIFICATION").ok().as_deref() == Some("1")
}

pub fn vulkan_device(suite: &str) -> Option<VulkanDevice> {
    if !VulkanDevice::probe() {
        if qualification_required() {
            panic!("{suite}: Vulkan device unavailable while KILN_QUALIFICATION=1");
        }
        eprintln!("{suite}: Vulkan device unavailable; skipping developer-only hardware test");
        return None;
    }

    match VulkanDevice::new() {
        Ok(device) => Some(device),
        Err(error) => {
            if qualification_required() {
                panic!(
                    "{suite}: Vulkan device initialization failed while KILN_QUALIFICATION=1: {error:#}"
                );
            }
            eprintln!(
                "{suite}: Vulkan device initialization failed; skipping developer-only hardware test: {error:#}"
            );
            None
        }
    }
}

#[allow(dead_code)]
pub fn vulkan_device_arc(suite: &str) -> Option<Arc<VulkanDevice>> {
    vulkan_device(suite).map(Arc::new)
}
