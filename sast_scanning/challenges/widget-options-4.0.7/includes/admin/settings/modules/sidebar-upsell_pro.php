<?php

/**
 * Purchase Key Validation Sidebar Metabox
 * Settings > Widget Options
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       4.0
 */

// Exit if accessed directly
if (!defined('ABSPATH')) exit;

/**
 * Create Metabox for Purchase Validation
 *
 * @since 4.0
 * @return void
 */
if (!function_exists('widgetopts_settings_upgrade_pro')) :
	function widgetopts_settings_upgrade_pro()
	{ ?>
		<div id="widgetopts-sidebar-widget-support" class="postbox widgetopts-sidebar-widget" style="border-color: #ffb310; border-width: 3px;">
			<h3 class="hndle ui-sortable-handle"><span><?php _e('Get Widget Options Extended', 'widget-options'); ?></span></h3>
			<div class="inside">
				<p>
					<?php
					$site_url = apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE));
					_e('<strong>Unlock all features!</strong> Get the world\'s most complete widget management and get best out of your widgets with <a href="' . $site_url . '" target="_blank">Widget Options Extended</a> including: ', 'widget-options'); ?>
				</p>
				<ul style="list-style: outside; padding-left: 15px;">
					<li>
						<?php _e('Custom Styling', 'widget-options'); ?>
					</li>
					<li>
						<?php _e('Widget Animations', 'widget-options'); ?>
					</li>
					<li>
						<?php _e('Custom Columns Display', 'widget-options'); ?>
					</li>
					<li>
						<?php _e('Shortcodes', 'widget-options'); ?>
					</li>
					<li>
						<?php _e('Taxonomies and Custom Post Types Support', 'widget-options'); ?>
					</li>
					<li>
						<?php _e('Fixed Sticky Widgets', 'widget-options'); ?>
					</li>
					<li>
						<?php _e('Widget Caching', 'widget-options'); ?>
					</li>
					<li>
						<?php _e('Clone Widgets', 'widget-options'); ?>
					</li>
					<li>
						<?php _e('and more pro-only features', 'widget-options'); ?>
					</li>
				</ul>
				<p>
					<a class="button-primary" href="<?php echo apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE) . '?utm_source=wordpressadmin&amp;utm_medium=widget&amp;utm_campaign=widgetoptsprocta'); ?>" target="_blank"><?php _e('Get Widget Options Extended', 'widget-options'); ?></a>
				</p>
			</div>
		</div>
<?php
	}
	add_action('widgetopts_module_sidebar', 'widgetopts_settings_upgrade_pro', 10);
endif;

?>